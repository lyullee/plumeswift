"""Replaceable building closures with explicit native call order and state."""
from dataclasses import dataclass, field
from typing import Protocol


class BuildingClosure(Protocol):
    def apply(self, workspace, ordered_buildings, context): ...


@dataclass
class BuildingContext:
    wakes: dict = field(default_factory=dict)
    max_velocity: float = 0.
    ordered_buildings: tuple = ()
    allow_unset_wake: bool = False

    def wake_state(self, number, error):
        wake = self.wakes.get(number)
        if wake is None and not self.allow_unset_wake:
            raise ValueError(error)
        return wake


@dataclass(frozen=True)
class UpwindClosure:
    mode: int = 2
    native_math: bool = False
    def apply(self, workspace, ordered_buildings, context):
        from .upwind import upwind_cavity
        for _, building in ordered_buildings:
            upwind_cavity(workspace, building, self.mode, native_math=self.native_math)


@dataclass(frozen=True)
class WakeClosure:
    mode: int = 2
    rooftop_mode: int = 2
    high_rise: bool = False
    native_math: bool = False
    def apply(self, workspace, ordered_buildings, context):
        from .wake import polygon_wake
        for number, building in ordered_buildings:
            context.wakes[number] = polygon_wake(workspace, building, number, mode=self.mode,
                rooftop_mode=self.rooftop_mode, high_rise=self.high_rise, native_math=self.native_math)


@dataclass(frozen=True)
class SidewallClosure:
    roughness: float = .01
    native_math: bool = False
    def apply(self, workspace, ordered_buildings, context):
        from .sidewall import sidewall
        for number, building in ordered_buildings:
            wake = context.wake_state(number, 'Sidewall closure requires the native polygon-wake geometry stage')
            frame = sidewall(workspace, building, wake, roughness=self.roughness, native_math=self.native_math)
            if wake is not None:
                wake.mixing_frame = frame


@dataclass(frozen=True)
class RooftopClosure:
    mode: int = 2
    roughness: float = .01
    native_math: bool = False
    def apply(self, workspace, ordered_buildings, context):
        from .rooftop import rooftop
        for number, building in ordered_buildings:
            wake = context.wake_state(number, 'Rooftop closure requires the native polygon-wake geometry stage')
            result = rooftop(workspace, building, wake, mode=self.mode, roughness=self.roughness,
                             native_math=self.native_math)
            if result is not None and wake is not None:
                wake.mixing_frame, angles = result
                if angles is not None:
                    wake.mixing_angles = angles


@dataclass(frozen=True)
class StreetCanyonClosure:
    roughness: float = .01
    native_math: bool = False
    def apply(self, workspace, ordered_buildings, context):
        from .street_canyon import street_canyon
        buildings_by_id = dict(ordered_buildings)
        for number, building in ordered_buildings:
            wake = context.wake_state(number, 'Street-canyon closure requires polygon-wake state')
            frame, angles = street_canyon(
                workspace, building, wake, buildings_by_id, context.max_velocity,
                roughness=self.roughness, native_math=self.native_math)
            if wake is not None:
                wake.mixing_frame, wake.mixing_angles = frame, angles


@dataclass(frozen=True)
class ModifiedStreetCanyonClosure:
    roughness: float = .01
    native_math: bool = False
    def apply(self, workspace, ordered_buildings, context):
        from .street_canyon import street_canyon
        buildings_by_id = dict(ordered_buildings)
        for number, building in ordered_buildings:
            wake = context.wake_state(number, 'Modified street canyon requires polygon-wake state')
            frame, angles = street_canyon(
                workspace, building, wake, buildings_by_id, context.max_velocity,
                roughness=self.roughness, mode=2, native_math=self.native_math)
            if wake is not None:
                wake.mixing_frame, wake.mixing_angles = frame, angles


@dataclass(frozen=True)
class BuildingPipeline:
    stages: tuple[BuildingClosure, ...]
    allow_unset_wake: bool = False
    def apply(self, workspace, ordered_buildings):
        import numpy as np
        maximum = 0. if workspace is None else float(1.2*np.hypot(workspace.arrays['u0'][-2], workspace.arrays['v0'][-2]).max())
        if workspace is not None and any(isinstance(stage, (StreetCanyonClosure, ModifiedStreetCanyonClosure)) and stage.native_math
                                         for stage in self.stages):
            peak = np.hypot(*(workspace.arrays[key][-2].astype(np.float32).astype(float) for key in ('u0', 'v0'))).max()
            maximum = float(np.float32(1.2*float(np.float32(peak))))
        context = BuildingContext(max_velocity=maximum, ordered_buildings=tuple(ordered_buildings),
                                  allow_unset_wake=self.allow_unset_wake)
        for stage in self.stages:
            stage.apply(workspace, context.ordered_buildings, context)
        return context

    @classmethod
    def native_order(cls, upwind=2, wake=2, sidewall=1, rooftop=2, high_rise=False, roughness=.01, street_canyon=0,
                     *, native_wake_math=False, native_building_math=False, allow_unset_wake=False):
        stages = []
        if upwind:
            stages.append(UpwindClosure(upwind, native_math=native_building_math))
        if wake:
            stages.append(WakeClosure(wake, rooftop, high_rise, native_math=native_wake_math or native_building_math))
        if street_canyon == 1:
            stages.append(StreetCanyonClosure(roughness, native_math=native_building_math))
        elif street_canyon == 2:
            stages.append(ModifiedStreetCanyonClosure(roughness, native_math=native_building_math))
        elif street_canyon:
            raise ValueError('Street canyon mode must be 0, 1 or 2')
        if sidewall:
            stages.append(SidewallClosure(roughness, native_math=native_building_math))
        if rooftop:
            stages.append(RooftopClosure(rooftop, roughness, native_math=native_building_math))
        return cls(tuple(stages), allow_unset_wake=allow_unset_wake)
