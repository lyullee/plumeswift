"""Progress clock shared by the native-style plume entry points."""
from datetime import timedelta


def write_legacy_options(stream, *, plume, wind, turbulence, basename, verbose, particle_output):
    """Report the selected standalone files before opening them."""
    divider = '-'*28
    lines = [
        'Summary of QES PLUME options: ', divider, 'Plume model in stand alone mode',
        f'Verbose:\t\t {"ON" if verbose else "OFF"}', divider, f'qesPlumeParamFile set to {plume}', divider,
        f'Winds input file set:        {wind}', f'Turbulence input file set:   {turbulence}', divider,
    ]
    if basename:
        lines.extend((divider, f'Output file basename:        {basename}', 'Plume output:                ON',
                      f'Plume particle output:       {"ON" if particle_output else "OFF"}'))
    print(*lines, '#'*67, sep='\n', file=stream, flush=True)


def write_legacy_particle_summary(stream, origin, final_ms, models):
    """Report terminal native counts from the same data as the JSON result."""
    divider = '-'*65+' '
    stamp = (origin+timedelta(milliseconds=final_ms)).isoformat()
    released = sum(model['released_particles'] for model in models.values())
    active = sum(model['active_particles'] for model in models.values())
    rogue = sum(model.get('rogue_particles', 0) for model in models.values())
    counts = [f'\t\t Name: {tag} with {model["active_particles"]} particles ' for tag, model in models.items()]
    print(divider, '[QES-Plume]\t End run particle summary ', f'\t\t Current simulation time: {stamp}',
          f'\t\t Simulation run time: {final_ms/1000:.6g}', f'\t\t Total number of particles released: {released}',
          f'\t\t Current number of particles in simulation: {active}', f'\t\t Number of rogue particles: {rogue}',
          *counts, divider, sep='\n', file=stream, flush=True)


class PlumeProgress:
    def __init__(self, origin, period_ms, stream):
        self.origin = origin
        self.period_ms = period_ms
        self.stream = stream
        self.iteration = 0

    def begin(self, start_ms):
        self.next_ms = start_ms+self.period_ms

    def _timestamp(self, time_ms):
        return (self.origin+timedelta(milliseconds=time_ms)).isoformat()

    @staticmethod
    def _counts(models):
        simulations = [model.simulation for model in models.values()]
        return (sum(simulation.released_particles for simulation in simulations),
                sum(len(simulation.batch.mass) for simulation in simulations))

    def begin_interval(self, time_ms, end_ms, models):
        self.begin(time_ms)
        released, active = self._counts(models)
        print(f'[QES-Plume]\t Advecting particles from {self._timestamp(time_ms)} to {self._timestamp(end_ms)}.',
              f'\t\t Total run time = {(end_ms-time_ms)/1000:.6g} s '
              f'(sim time = {time_ms/1000:.6g} s, iteration = {self.iteration}). ',
              f'\t\t Particles: Released = {released} Active = {active}.', sep='\n', file=self.stream, flush=True)

    def end_interval(self, time_ms, models):
        released, active = self._counts(models)
        print(f'[QES-Plume]\t End of particles advection at Time = {self._timestamp(time_ms)} '
              f's (iteration = {self.iteration}). ', f'\t\t Particles: Released = {released} Active = {active}.',
              sep='\n', file=self.stream, flush=True)

    def step(self, time_ms, end_ms, models):
        self.iteration += 1
        if time_ms < self.next_ms and time_ms != end_ms:
            return
        released, active = self._counts(models)
        stamp = self._timestamp(time_ms)
        print(f'[QES-Plume]\t Time = {stamp} (t = {time_ms/1000:.6g} s, iter = {self.iteration}). '
              f'Particles: released = {released} active = {active}.', file=self.stream, flush=True)
        self.next_ms += self.period_ms
