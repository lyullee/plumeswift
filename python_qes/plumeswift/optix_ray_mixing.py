"""Trace method-3 mixing rays with NVIDIA OptiX via its Python host bindings."""
from contextlib import ExitStack
import os
from pathlib import Path
import struct
import sys

import numpy as np

from .numerics.cuda_runtime import load_cupy


def load_optix():
    """Load the optional host binding before creating a CUDA context."""
    if os.name == 'nt' and 'cupy' in sys.modules and 'optix' not in sys.modules:
        raise RuntimeError('Initialize NVIDIA PyOptiX before importing CuPy on Windows')
    try:
        with ExitStack() as paths:
            if os.name == 'nt' and (toolkit := os.environ.get('CUDA_PATH')):
                dll_path = Path(toolkit)/'bin'
                if dll_path.is_dir():
                    paths.enter_context(os.add_dll_directory(str(dll_path)))
            import optix
    except ImportError as exc:
        raise RuntimeError('OptiX mixing requires the NVIDIA PyOptiX bindings built for the installed driver') from exc
    if optix.version()[:2] < (7, 7):
        raise RuntimeError('OptiX mixing requires NVIDIA PyOptiX built against OptiX 7.7 or newer')
    return optix


def _distance_program():
    """Emit a ray-generation program using the OptiX device PTX intrinsics."""
    # The trace intrinsic has 32 payload slots. Only slot zero carries distance.
    payload_out = ','.join(f'%payload{i}' for i in range(32))
    payload_in = ','.join(['%infinity_bits']+['%zero']*31)
    return f'''
.version 7.8
.target sm_60
.address_size 64
.const .align 8 .b8 params[40];

.visible .entry __raygen__distance()
{{
    .reg .pred %done;
    .reg .b32 %point, %rays, %ray, %zero, %one, %mask, %infinity_bits;
    .reg .b32 %payload<32>;
    .reg .b64 %scene, %origins, %directions, %output, %offset, %address;
    .reg .f32 %ox, %oy, %oz, %dx, %dy, %dz, %best, %hit, %limit, %zero_f;

    call (%point), _optix_get_launch_index_x, ();
    ld.const.u64 %scene, [params];
    ld.const.u64 %origins, [params+8];
    ld.const.u64 %directions, [params+16];
    ld.const.u64 %output, [params+24];
    ld.const.u32 %rays, [params+32];
    mul.wide.u32 %offset, %point, 12;
    add.u64 %address, %origins, %offset;
    ld.global.f32 %ox, [%address];
    ld.global.f32 %oy, [%address+4];
    ld.global.f32 %oz, [%address+8];
    mov.u32 %zero, 0;
    mov.u32 %one, 1;
    mov.u32 %mask, 255;
    mov.u32 %infinity_bits, 0x7f800000;
    mov.f32 %zero_f, 0f00000000;
    mov.f32 %best, 0f7F800000;
    mov.u32 %ray, 0;
next_ray:
    setp.ge.u32 %done, %ray, %rays;
    @%done bra store_result;
    ld.global.f32 %dx, [%directions];
    ld.global.f32 %dy, [%directions+4];
    ld.global.f32 %dz, [%directions+8];
    min.f32 %limit, %best, 0f5A0E1BCA;
    call ({payload_out}), _optix_trace_typed_32,
        (%zero,%scene,%ox,%oy,%oz,%dx,%dy,%dz,%zero_f,%limit,%zero_f,%mask,%one,
         %zero,%one,%zero,%one,{payload_in});
    mov.b32 %hit, %payload0;
    min.f32 %best, %best, %hit;
    add.u64 %directions, %directions, 12;
    add.u32 %ray, %ray, 1;
    bra next_ray;
store_result:
    mul.wide.u32 %offset, %point, 4;
    add.u64 %address, %output, %offset;
    st.global.f32 [%address], %best;
    ret;
}}

.visible .entry __miss__distance()
{{
    .reg .b32 %slot, %miss;
    mov.u32 %slot, 0;
    mov.u32 %miss, 0x7f800000;
    call _optix_set_payload, (%slot, %miss);
    ret;
}}

.visible .entry __closesthit__distance()
{{
    .reg .f32 %distance;
    .reg .b32 %slot, %bits;
    mov.u32 %slot, 0;
    call (%distance), _optix_get_ray_tmax, ();
    mov.b32 %bits, %distance;
    call _optix_set_payload, (%slot, %bits);
    ret;
}}
'''.encode('ascii')


def _pipeline(optix, context, resources, program, entries):
    options = optix.PipelineCompileOptions(
        traversableGraphFlags=optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
        numPayloadValues=1, numAttributeValues=2, pipelineLaunchParamsVariableName='params',
        usesPrimitiveTypeFlags=optix.PRIMITIVE_TYPE_FLAGS_TRIANGLE)
    module, _ = context.moduleCreate(optix.ModuleCompileOptions(), options, program)
    resources.callback(module.destroy)
    raygen, miss, hit = optix.ProgramGroupDesc(), optix.ProgramGroupDesc(), optix.ProgramGroupDesc()
    raygen.raygenModule, raygen.raygenEntryFunctionName = module, entries[0]
    miss.missModule, miss.missEntryFunctionName = module, entries[1]
    hit.hitgroupModuleCH, hit.hitgroupEntryFunctionNameCH = module, entries[2]
    groups, _ = context.programGroupCreate([raygen, miss, hit])
    for group in groups:
        resources.callback(group.destroy)
    link = optix.PipelineLinkOptions(maxTraceDepth=1)
    pipeline = context.pipelineCreate(options, link, groups, '')
    resources.callback(pipeline.destroy)
    stack = optix.StackSizes()
    for group in groups:
        optix.util.accumulateStackSizes(group, stack, pipeline)
    pipeline.setStackSize(*optix.util.computeStackSizes(stack, 1, 0, 0), 1)
    return pipeline, groups


def _records(optix, cp, groups):
    size = optix.SBT_RECORD_HEADER_SIZE
    host = np.zeros((len(groups), size), dtype=np.uint8)
    for group, record in zip(groups, host):
        optix.sbtRecordPackHeader(group, record)
    device = cp.asarray(host)
    ptr = device.data.ptr
    table = optix.ShaderBindingTable(
        raygenRecord=ptr, missRecordBase=ptr+size, missRecordStrideInBytes=size, missRecordCount=1,
        hitgroupRecordBase=ptr+2*size, hitgroupRecordStrideInBytes=size, hitgroupRecordCount=1)
    return table, device


def _acceleration(optix, cp, context, stream, vertices):
    source = optix.BuildInputTriangleArray()
    source.vertexFormat = optix.VERTEX_FORMAT_FLOAT3
    source.numVertices = vertices.size//3
    source.vertexBuffers = [vertices.data.ptr]
    source.flags = [optix.GEOMETRY_FLAG_DISABLE_ANYHIT]
    source.numSbtRecords = 1
    options = optix.AccelBuildOptions(buildFlags=optix.BUILD_FLAG_PREFER_FAST_TRACE,
                                      operation=optix.BUILD_OPERATION_BUILD)
    sizes = context.accelComputeMemoryUsage([options], [source])
    scratch = cp.cuda.alloc(sizes.tempSizeInBytes)
    geometry = cp.cuda.alloc(sizes.outputSizeInBytes)
    handle = context.accelBuild(stream.ptr, [options], [source], scratch.ptr, sizes.tempSizeInBytes,
                                geometry.ptr, sizes.outputSizeInBytes, [])
    stream.synchronize()
    return handle, geometry


_MAX_RAYS_PER_LAUNCH = 1 << 22


def optix_ray_distances(mesh, points, directions, *, device=0):
    """Return one nearest-hit distance per point, using +infinity for misses."""
    points = np.asarray(points, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    if (points.ndim != 2 or points.shape[1] != 3 or directions.ndim != 2 or directions.shape[1] != 3
            or not np.isfinite(points).all() or not np.isfinite(directions).all()):
        raise ValueError('Expected finite N by 3 OptiX ray origins and directions')
    if not len(points) or not len(directions) or not len(mesh.triangles):
        return np.full(len(points), np.inf, dtype=np.float32)
    optix = load_optix()
    cp = load_cupy(device)
    with cp.cuda.Device(device), ExitStack() as resources:
        # Recent bindings initialize OptiX on import. Older builds expose init().
        initialize = getattr(optix, 'init', None)
        if initialize is not None:
            initialize()
        vertices = cp.asarray(mesh.triangles, dtype=cp.float32, order='C')
        current = cp.cuda.driver.ctxGetCurrent()
        if not current:
            cp.cuda.runtime.free(0)
            current = cp.cuda.driver.ctxGetCurrent()
        context = optix.deviceContextCreate(current, optix.DeviceContextOptions())
        resources.callback(context.destroy)
        entries = ('__raygen__distance', '__miss__distance', '__closesthit__distance')
        pipeline, groups = _pipeline(optix, context, resources, _distance_program(), entries)
        stream = cp.cuda.get_current_stream()
        handle, geometry = _acceleration(optix, cp, context, stream, vertices)
        sbt, records = _records(optix, cp, groups)
        origins = cp.asarray(points, order='C')
        rays = cp.asarray(directions, order='C')
        distances = cp.empty(len(points), dtype=cp.float32)
        # Each OptiX thread traces all directions. A point-only limit is
        # too large when a production case uses thousands of rays per cell.
        tile = min(1 << 20, max(1, _MAX_RAYS_PER_LAUNCH//len(directions)))
        for start in range(0, len(points), tile):
            count = min(len(points)-start, tile)
            block = struct.pack('<QQQQI4x', handle, origins[start:].data.ptr, rays.data.ptr,
                                distances[start:].data.ptr, len(directions))
            params = cp.asarray(np.frombuffer(block, dtype=np.uint8))
            optix.launch(pipeline, stream.ptr, params.data.ptr, len(block), sbt, count, 1, 1)
            stream.synchronize()
        return cp.asnumpy(distances)
