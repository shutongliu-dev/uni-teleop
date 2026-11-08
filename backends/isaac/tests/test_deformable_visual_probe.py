import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.validation.deformable_visual_probe import rigid_residual, _world_points, _require_current_output, _fabric_points


def test_visual_audit_removes_arbitrary_base_motion_and_detects_compression():
    rest = np.array([[-.05, -.03, .004], [.05, .03, .004], [0., 0., .045]])
    rot = Rotation.from_euler('xyz', [.1, -.2, .4])
    pose = np.r_[[.6, -.8, .9], rot.as_quat()]
    actual = rot.apply(rest)+pose[:3]
    assert rigid_residual(actual, rest, pose)['max_nonrigid_displacement_m'] < 1e-12
    actual[-1] -= [.0, .0, .004]
    metrics = rigid_residual(actual, rest, pose)
    assert np.isclose(metrics['max_nonrigid_displacement_m'], .004)
    assert np.isclose(metrics['minimum_nonrigid_vertical_displacement_m'], -.004)
    matrix = np.eye(4)
    matrix[:3, :3] = rot.as_matrix().T
    matrix[3, :3] = pose[:3]
    np.testing.assert_allclose(_world_points(rest, matrix), rot.apply(rest)+pose[:3])


def test_active_fabric_transport_requires_its_own_current_points():
    assert _require_current_output(fabric_renderer=True, usd_error=.01, fabric_error=1e-8) == 'Fabric'
    with pytest.raises(RuntimeError, match='Fabric'):
        _require_current_output(fabric_renderer=True, usd_error=0., fabric_error=.01)
    with pytest.raises(RuntimeError, match='Fabric'):
        _require_current_output(fabric_renderer=True, usd_error=0., fabric_error=None)
    with pytest.raises(RuntimeError, match='USD'):
        _require_current_output(fabric_renderer=False, usd_error=.01, fabric_error=0.)


def test_fabric_reader_requests_only_read_access_and_owned_copy(monkeypatch):
    import sys
    from types import SimpleNamespace
    from scripts.validation import deformable_visual_probe as probe
    actual = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    calls = []
    class Tracker:
        def __init__(self, stage):calls.append('tracker')
        def TrackAttribute(self, name):calls.append(('track',name))
        def ClearChanges(self):calls.append('clear_own_listener')
        def GetAllChangedAttributes(self):return []
    class Stage:
        def SelectPrims(self, **kwargs):
            assert kwargs['require_attrs']==[('point3farray','points','READ'),('matrix4d','omni:fabric:worldMatrix','READ')]
            assert kwargs['device']=='cuda:0' and kwargs['want_paths'] is True
            return SimpleNamespace(GetPaths=lambda:['/Other','/Foam'])
        def GetPrimAtPath(self, path):
            assert path=='/Foam'
            return SimpleNamespace(HasAttribute=lambda name:name=='Deformable')
    monkeypatch.setitem(sys.modules,'usdrt',SimpleNamespace(Rt=SimpleNamespace(ChangeTracker=Tracker),
        Sdf=SimpleNamespace(ValueTypeNames=SimpleNamespace(Point3fArray='point3farray',Matrix4d='matrix4d')),
        Usd=SimpleNamespace(Access=SimpleNamespace(Read='READ'))))
    def copy(selection,row,count,device):
        assert row==1 and count==2 and device=='cuda:0'
        return actual.copy(),np.eye(4)
    monkeypatch.setattr(probe,'_copy_fabric_selection',copy)
    points,world,receipt=_fabric_points(Stage(),'/Foam','cuda:0',2)
    np.testing.assert_array_equal(points,actual);np.testing.assert_array_equal(world,actual)
    assert receipt['requested_access']=='Read' and receipt['tracked_attribute_changes']==[]
    assert calls==['tracker',('track','points'),('track','omni:fabric:worldMatrix'),'clear_own_listener']
    Tracker.GetAllChangedAttributes=lambda self:['/Foam.points']
    with pytest.raises(RuntimeError,match='marked tracked attributes changed'):
        _fabric_points(Stage(),'/Foam','cuda:0',2)


def test_fabric_point_reader_has_no_vt_array_or_numpy_source_export():
    import ast,inspect
    from scripts.validation import deformable_visual_probe as probe
    tree=ast.parse(inspect.getsource(probe._fabric_points))
    calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
    assert 'Get' not in calls and 'DetachFromSource' not in calls
    assert 'SelectPrims' in calls and 'GetAllChangedAttributes' in calls


def test_fabric_flush_uses_fixed_sdk_dt_before_time_contract():
    from scripts.validation.deformable_visual_probe import _flush_native_fabric
    class FixedSdkContract:
        def force_update(self, dt, current_time):
            assert dt == .0005
            assert current_time == 1.601
    receipt = _flush_native_fabric(FixedSdkContract(), physics_dt=.0005, current_time=1.601)
    assert receipt['positional_argument_order'] == ['physics_dt_s', 'current_time_s']
    with pytest.raises(ValueError, match='positive dt'):
        _flush_native_fabric(FixedSdkContract(), physics_dt=0., current_time=1.601)


def test_owned_read_copy_kernel_on_cpu_fabric_fixture():
    wp=pytest.importorskip('warp')
    wp.config.enable_cuda=False  # This tiny compatibility fixture never launches CUDA.
    from warp.tests.test_fabricarray import _create_fabric_array_array_interface,_create_fabric_array_interface
    from scripts.validation.deformable_visual_probe import _copy_fabric_selection
    expected=np.array([[1.,2.,3.],[4.,5.,6.]],dtype=np.float32)
    points=wp.array(expected,dtype=wp.vec3f,device='cpu')
    matrices=wp.array(np.eye(4)[None],dtype=wp.mat44d,device='cpu')
    p=_create_fabric_array_array_interface([points],'points',bucket_sizes=[1])
    m=_create_fabric_array_interface(matrices,'omni:fabric:worldMatrix',bucket_sizes=[1])
    interface={**p,'attribs':{**p['attribs'],**m['attribs']}}
    for spec in interface['attribs'].values():spec['access']=0  # USDRT Access.Read
    result,matrix=_copy_fabric_selection(interface,0,2,'cpu')
    np.testing.assert_array_equal(result,expected);np.testing.assert_array_equal(matrix,np.eye(4))
    np.testing.assert_array_equal(points.numpy(),expected)
    with pytest.raises(RuntimeError,match='point count'):_copy_fabric_selection(interface,0,3,'cpu')
