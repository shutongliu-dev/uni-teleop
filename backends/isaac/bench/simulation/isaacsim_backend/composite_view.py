"""One logical robot made of several PhysX articulations (bimanual FR3 + hands).

PhysX caps an articulation at 64 links, so each arm is its own articulation.
This adapter presents the exact ArticulationView subset the env uses, with DOF
and link axes concatenated in view order; setters split their tensor back per
articulation. Root setters take one row per articulation. Import only inside a
running SimulationApp (it uses torch tensors from the native views).
"""
from types import SimpleNamespace


class CompositeArticulationView:
    def __init__(self, views):
        if not views or any(view.count != 1 for view in views):
            raise RuntimeError('Every composite member must be one native articulation')
        self.views = list(views)
        self.dof_counts = [len(view.shared_metatype.dof_names) for view in self.views]
        self.link_counts = [len(view.shared_metatype.link_names) for view in self.views]
        names = [name for view in self.views for name in view.shared_metatype.dof_names]
        links = [name for view in self.views for name in view.shared_metatype.link_names]
        if len(set(names)) != len(names) or len(set(links)) != len(links):
            raise RuntimeError('Composite articulation names must be unique across members')
        self.shared_metatype = SimpleNamespace(dof_names=names, link_names=links)
        self.count = 1

    def _cat(self, method):
        import torch
        return torch.cat([getattr(view, method)() for view in self.views], dim=1)

    def _split(self, method, value, indices, counts):
        if len(value.shape) != 2 or value.shape[1] != sum(counts):
            raise ValueError(f'{method}: tensor shape {value.shape} must have width {sum(counts)}')
        offset = 0
        for view, count in zip(self.views, counts):
            getattr(view, method)(value[:, offset:offset + count].contiguous(), indices)
            offset += count

    # DOF-wise state and parameters
    def get_dof_positions(self): return self._cat('get_dof_positions')
    def get_dof_limits(self): return self._cat('get_dof_limits')
    def get_dof_velocities(self): return self._cat('get_dof_velocities')
    def get_dof_stiffnesses(self): return self._cat('get_dof_stiffnesses')
    def get_dof_dampings(self): return self._cat('get_dof_dampings')
    def get_dof_armatures(self): return self._cat('get_dof_armatures')
    def get_dof_friction_coefficients(self): return self._cat('get_dof_friction_coefficients')
    def get_dof_friction_properties(self): return self._cat('get_dof_friction_properties')
    def get_dof_max_velocities(self): return self._cat('get_dof_max_velocities')
    def get_dof_max_forces(self): return self._cat('get_dof_max_forces')

    def get_gravity_compensation_forces(self):
        import torch
        parts = []
        for view, count in zip(self.views, self.dof_counts):
            forces = view.get_gravity_compensation_forces()
            parts.append(forces[:, forces.shape[1] - count:])
        return torch.cat(parts, dim=1)

    def set_dof_positions(self, value, indices): self._split('set_dof_positions', value, indices, self.dof_counts)
    def set_dof_velocities(self, value, indices): self._split('set_dof_velocities', value, indices, self.dof_counts)
    def set_dof_position_targets(self, value, indices): self._split('set_dof_position_targets', value, indices, self.dof_counts)
    def set_dof_velocity_targets(self, value, indices): self._split('set_dof_velocity_targets', value, indices, self.dof_counts)
    def set_dof_actuation_forces(self, value, indices): self._split('set_dof_actuation_forces', value, indices, self.dof_counts)
    def set_dof_stiffnesses(self, value, indices): self._split('set_dof_stiffnesses', value, indices, self.dof_counts)
    def set_dof_dampings(self, value, indices): self._split('set_dof_dampings', value, indices, self.dof_counts)
    def set_dof_max_forces(self, value, indices): self._split('set_dof_max_forces', value, indices, self.dof_counts)
    def set_dof_armatures(self, value, indices): self._split('set_dof_armatures', value, indices, self.dof_counts)
    def set_dof_friction_coefficients(self, value, indices): self._split('set_dof_friction_coefficients', value, indices, self.dof_counts)
    def set_dof_friction_properties(self, value, indices): self._split('set_dof_friction_properties', value, indices, self.dof_counts)
    def set_dof_max_velocities(self, value, indices): self._split('set_dof_max_velocities', value, indices, self.dof_counts)

    # Link-wise state and properties
    def get_link_transforms(self): return self._cat('get_link_transforms')
    def get_link_velocities(self): return self._cat('get_link_velocities')
    def get_masses(self): return self._cat('get_masses')

    def get_material_properties(self):
        return self._cat('get_material_properties')

    # One root row per member articulation, in view order.
    def set_root_transforms(self, value, indices):
        if value.shape[0] != len(self.views):
            raise ValueError('Composite root transforms need one row per articulation')
        for row, view in enumerate(self.views):
            view.set_root_transforms(value[row:row + 1].contiguous(), indices)

    def set_root_velocities(self, value, indices):
        if value.shape[0] != len(self.views):
            raise ValueError('Composite root velocities need one row per articulation')
        for row, view in enumerate(self.views):
            view.set_root_velocities(value[row:row + 1].contiguous(), indices)
