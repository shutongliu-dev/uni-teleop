"""Native Isaac Agent API. Only server-projected observations cross this boundary."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response

from bench.paths import OUTPUT_ROOT, load_json
from scenes.manipulation import DEFAULT_TASK_RELEASE
from scenes.manipulation.contracts import project_grasp_contract
from bench.protocol.schemas import EvaluationPlan
from bench.simulation.isaacsim_backend.robot_registry import list_hands
from scripts.maintenance.freeze_project_sources import verify_if_snapshot
from .service import AgentService, CreateEpisodeRequest, CreateBoardGameRequest, CreateTurnGameRequest, ServiceError
from .robot_support import IKQuery
from bench.protocol.functions import FunctionCall, FunctionCatalog, FunctionResponse, function_schemas
from .functions import public_failure


def create_app(data_dir=None, *, service=None, **service_options):
    directory = data_dir or os.environ.get('BENCH_AGENT_DATA_DIR') or Path(OUTPUT_ROOT)/'agent_bench/episodes'
    if service is None and os.environ.get('BENCH_EVALUATION_PLAN') and 'evaluation_plan' not in service_options:
        service_options['evaluation_plan']=EvaluationPlan.model_validate_json(Path(os.environ['BENCH_EVALUATION_PLAN']).read_text())
    service_options.setdefault('device',os.environ.get('BENCH_DEVICE','cuda:0'))
    service_options.setdefault('release',Path(os.environ.get('BENCH_TASK_RELEASE',str(DEFAULT_TASK_RELEASE))))
    manager = service or AgentService(directory, **service_options)
    source_verification = verify_if_snapshot(Path(__file__).absolute().parents[2],
        required_modules=("bench.runtime.server", "bench.runtime.service"),
        entrypoint_file=Path(__file__))

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(manager.shutdown)

    app=FastAPI(title='Dex Agent Bench — native Isaac Sim',version='0.4.0',lifespan=lifespan)
    app.state.manager=manager
    app.state.source_snapshot_verification=source_verification

    @app.exception_handler(ServiceError)
    async def service_error(request,error):
        if request.url.path.endswith('/call') or request.url.path.startswith(('/board-games/', '/turn-games/')):
            value=public_failure(code=error.code,message=str(error),status=error.status)
            return JSONResponse(status_code=error.status,content=value.model_dump(mode='json'))
        return JSONResponse(status_code=error.status,content={'detail':str(error)})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request,error):
        if request.url.path.endswith('/call'):
            try:
                session=manager.get(request.path_params['episode_id'])
            except ServiceError as exc:
                value=public_failure(code=exc.code,message=str(exc),status=exc.status)
            else:
                # Validation failures are still bench motion attempts when a
                # known action function was requested; charge and record once.
                value=await asyncio.to_thread(session.functions.invoke,error.body)
            return JSONResponse(status_code=value.error.http_status,content=value.model_dump(mode='json'))
        return JSONResponse(status_code=422,content={'detail':'Request violates the advertised contract'})

    @app.exception_handler(Exception)
    async def internal_error(request,error):
        logging.getLogger(__name__).error('Private service failure',exc_info=error)
        if request.url.path.endswith('/call'):
            value=public_failure(code='infrastructure_error',message='Bench function failed; inspect operator diagnostics.',status=503)
            return JSONResponse(status_code=503,content=value.model_dump(mode='json'))
        return JSONResponse(status_code=503,content={'detail':'Native service failed; operator diagnostics are private'})

    @app.get('/health')
    async def health():
        return {'status':'ok','protocol':'agent_bench_v1','backend':'isaacsim_native_physx',
                'physics_readiness':'checked when an episode is created','hardware_connected':False}

    @app.get('/tasks')
    def tasks():
        root=manager.release
        from bench.isaacsim.release import validate
        manifest=validate(root,assets=False)
        output=[]
        for entry in manifest['tasks']:
            task=load_json(root/entry['task'])
            if task.get('schema_version') == 'living-room-manipulation-task-v1':
                from .manipulation_tasks import project_manipulation_task
                output.append(project_manipulation_task(task))
                continue
            if task.get('schema_version') == 'card-room-placement-task-v1':
                from .placement_tasks import project_placement_task
                output.append(project_placement_task(task, episode=False))
                continue
            if entry.get('task_kind') == 'board_game':
                output.append({'task_id':entry['id'],'instruction':entry['title_zh'],
                    'task_kind':'board_game','create_endpoint':'POST /board-games',
                    'rules':task['rules'],'geometry':task['geometry'],
                    'public_rules':task['public_rules'],'public_rules_sha256':task['public_rules_sha256']})
                continue
            if entry.get('task_kind', manifest.get('task_kind')) == 'turn_game':
                output.append({'task_id':entry['id'], 'instruction':task['title_zh'],
                    'task_kind':'turn_game', 'create_endpoint':'POST /turn-games',
                    'public_rules':task['public_rules']})
                continue
            if task.get('public_goal', {}).get('operation') == 'cloth_table_drag':
                from .cloth_tasks import project_cloth_task
                output.append(project_cloth_task(entry['id'], task))
                continue
            from bench.protocol.task_referee import project_task_rules
            from scenes.runtime import terminal_scoring_contract
            referee_task = project_task_rules(task, scoring_contract=terminal_scoring_contract(task))
            if referee_task is not None:
                output.append(referee_task)
                continue
            row={'task_id':entry['id'],'instruction':task['title_zh']+'。'+
                 '；'.join(step['instruction_zh'] for step in task['steps'])}
            # Only a scene pinning a non-v1 hand mount publishes it (v1 rows are unchanged).
            scene_path=root/entry['scene'] if 'scene' in entry else None
            scene_robot=(load_json(scene_path).get('robot',{})
                         if scene_path is not None and scene_path.is_file() else {})
            if scene_robot.get('assembly_version'):
                from bench.simulation.isaacsim_backend.robot_registry import get_robot_spec
                mount_spec=get_robot_spec(scene_robot)
                if mount_spec['mount_convention']!='flange_axis_centred_v1':
                    row['robot_mount']={'mount_convention':mount_spec['mount_convention'],
                                        'hand_frame_yaw_about_flange_axis_rad':mount_spec['hand_frame_yaw_rad'],
                                        'status':mount_spec['mount_status']}
            if 'public_goal' in task:
                # Only declared static intent and success conditions are public.
                # Live evaluator evidence and native object state stay private.
                row['goal']={key:task['public_goal'][key] for key in
                    ('object_id','operation','target_position_m','frame_id','description') if key in task['public_goal']}
                if 'bimanual_success_contract' in task['public_goal']:
                    from scenes.manipulation.contracts import BimanualSuccessContract
                    row['goal']['bimanual_success_contract']=BimanualSuccessContract.model_validate(
                        task['public_goal']['bimanual_success_contract']).model_dump(mode='json')
                # Static scene facts published by newer releases (table height, frame,
                # target reference, nominal mass); older releases simply lack them.
                for key in ('workspace','target_reference','object_mass_kg'):
                    if key in task['public_goal']:
                        row['goal'][key]=task['public_goal'][key]
                if 'grasp_constraints' in task['public_goal']:
                    row['goal']['grasp_constraints']=project_grasp_contract(task['public_goal']['grasp_constraints'])
                if 'success_contract' in task['public_goal']:
                    from scenes.manipulation.contracts import project_success_contract
                    row['goal']['success_contract']=project_success_contract(task['public_goal']['success_contract'])
            output.append(row)
        return output

    @app.get('/robots')
    def robots():
        keys=('hand_id','label','actuated_joint_count','joint_count','hardware_actuator_equivalent','control_model','mount_status',
              'integration_status')
        return [{k:v for k,v in hand.items() if k in keys} for hand in list_hands()]

    def policy_descriptors(task_id):
        if task_id is not None:
            from bench.isaacsim.release import validate
            manifest = validate(manager.release, assets=False)
            if task_id not in {entry['id'] for entry in manifest['tasks']}:
                raise ServiceError(422, 'Unknown task ID')
        try:
            return manager.registry.descriptors(task_id=task_id)
        except ValueError:
            raise ServiceError(422, 'Task ID is required for task-bound controller discovery') from None

    @app.get('/policies')
    def policies(task_id: str | None = None):
        return [d.model_dump(mode='json') for d in policy_descriptors(task_id)]

    @app.get('/interfaces')
    def interfaces(task_id: str | None = None):
        from .catalog import board_game_interface_catalog, turn_game_interface_catalog, interface_catalog
        from .free_track import advertise_free_track
        descriptors=policy_descriptors(task_id)
        catalog=advertise_free_track(interface_catalog(descriptors),descriptors)
        if getattr(manager, 'release', None) is not None:
            from bench.isaacsim.release import validate
            manifest=validate(manager.release,assets=False)
            selected=[entry for entry in manifest['tasks'] if task_id is None or entry['id']==task_id]
            if selected and all(entry.get('task_kind', manifest.get('task_kind'))=='board_game'
                                for entry in selected):
                return board_game_interface_catalog(catalog)
            if selected and all(entry.get('task_kind', manifest.get('task_kind'))=='turn_game'
                                for entry in selected):
                return turn_game_interface_catalog(catalog)
        return catalog

    @app.post('/episodes',status_code=201)
    async def create_episode(request:CreateEpisodeRequest):
        return await asyncio.to_thread(manager.create,request)

    def bearer(request):
        value=request.headers.get('authorization','')
        return value[7:] if value.startswith('Bearer ') else None

    def player(request,episode_id,player_id):
        kind='turn_game' if request.url.path.startswith('/turn-games/') else 'board_game'
        return manager.player(episode_id,player_id,bearer(request),kind=kind)

    @app.get('/turn-games/schema')
    def turn_game_schema():
        from bench.protocol.turn_games import turn_game_schemas
        return {'protocol':'turn_game_state_v1', 'capability_version':'1.0.0',
                'schemas':{**turn_game_schemas(), 'CreateTurnGameRequest':CreateTurnGameRequest.model_json_schema()},
                'reference':None}

    @app.post('/turn-games',status_code=201)
    async def create_turn_game(request:CreateTurnGameRequest):
        return await asyncio.to_thread(manager.create,request,turn_game=True)

    @app.get('/board-games/schema')
    def board_game_schema():
        from bench.protocol.board_games import board_game_schemas
        return {'protocol':'bench_board_game_state_v1','capability_version':'1.0.0',
                'schemas':{**board_game_schemas(),'CreateBoardGameRequest':CreateBoardGameRequest.model_json_schema()},
                'reference':None}

    @app.post('/board-games',status_code=201)
    async def create_board_game(request:CreateBoardGameRequest):
        return await asyncio.to_thread(manager.create,request,board_game=True)

    @app.get('/turn-games/{episode_id}/players/{player_id}/functions')
    @app.get('/board-games/{episode_id}/players/{player_id}/functions')
    async def board_functions(episode_id:str,player_id:str,request:Request):
        return await asyncio.to_thread(player(request,episode_id,player_id).catalog,player_id)

    @app.get('/turn-games/{episode_id}/players/{player_id}/state')
    @app.get('/board-games/{episode_id}/players/{player_id}/state')
    async def board_state(episode_id:str,player_id:str,request:Request):
        return await asyncio.to_thread(player(request,episode_id,player_id).public,player_id)

    @app.post('/turn-games/{episode_id}/players/{player_id}/intent')
    @app.post('/board-games/{episode_id}/players/{player_id}/intent')
    async def board_intent(episode_id:str,player_id:str,request:Request):
        session=player(request,episode_id,player_id)
        try:
            payload=await request.json()
        except (ValueError,UnicodeError):
            payload=None
        return await asyncio.to_thread(session.intent,player_id,payload)

    @app.post('/turn-games/{episode_id}/players/{player_id}/call',response_model=FunctionResponse)
    @app.post('/board-games/{episode_id}/players/{player_id}/call',response_model=FunctionResponse)
    async def board_call(episode_id:str,player_id:str,request:Request):
        # Authentication precedes even malformed-call accounting. The ordinary
        # /episodes routes cannot provide an unscoped route into this ledger.
        session=player(request,episode_id,player_id)
        try:
            payload=await request.json()
        except (ValueError,UnicodeError):
            payload=None
        response=await asyncio.to_thread(session.invoke,player_id,payload)
        status=(response.error.http_status if not response.ok else 202
                if response.function in ('move_geometry','move_wrist','set_hand_targets','move_joints','hold')
                and response.result.status in ('accepted','running') else 200)
        return JSONResponse(status_code=status,content=response.model_dump(mode='json'))

    @app.get('/turn-games/{episode_id}/players/{player_id}/result')
    @app.get('/board-games/{episode_id}/players/{player_id}/result')
    async def board_result(episode_id:str,player_id:str,request:Request):
        return await asyncio.to_thread(player(request,episode_id,player_id).result,player_id)

    @app.get('/functions/schema')
    def callable_schema():
        return {'protocol':'bench_functions_v1','capability_version':'1.0.0',
                'schemas':function_schemas(),'reference':None}

    @app.get('/episodes/{episode_id}/functions',response_model=FunctionCatalog)
    async def episode_functions(episode_id:str):
        return await asyncio.to_thread(manager.get(episode_id).functions.catalog)

    @app.post('/episodes/{episode_id}/call',response_model=FunctionResponse)
    async def call_function(episode_id:str,payload:FunctionCall):
        response=await asyncio.to_thread(manager.get(episode_id).functions.invoke,payload)
        status=(response.error.http_status if not response.ok else 202
                if response.function in ('move_geometry','move_wrist','set_hand_targets','move_joints','hold')
                and response.result.status in ('accepted','running') else 200)
        return JSONResponse(status_code=status,content=response.model_dump(mode='json'))

    @app.get('/episodes/{episode_id}/observation')
    async def observe(episode_id:str):
        return await asyncio.to_thread(manager.get(episode_id).observe)

    @app.get('/episodes/{episode_id}/task')
    async def task_requirements(episode_id:str):
        return await asyncio.to_thread(manager.get(episode_id).task_requirements)

    @app.get('/episodes/{episode_id}/robot')
    async def robot_guide(episode_id:str):
        return await asyncio.to_thread(manager.get(episode_id).robot_guide)

    @app.post('/episodes/{episode_id}/robot/ik')
    async def query_ik(episode_id:str, query:IKQuery):
        return await asyncio.to_thread(manager.get(episode_id).query_ik, query)

    @app.post('/episodes/{episode_id}/actions',status_code=202)
    async def action(episode_id:str,request:Request):
        try:
            payload=await request.json()
        except (ValueError,UnicodeError):
            payload=None
        return await asyncio.to_thread(manager.get(episode_id).submit,payload)

    @app.get('/episodes/{episode_id}/actions/{action_id}')
    async def action_status(episode_id:str,action_id:str):
        return await asyncio.to_thread(manager.get(episode_id).status,action_id)

    @app.post('/episodes/{episode_id}/actions/{action_id}/cancel')
    async def cancel(episode_id:str,action_id:str):
        return await asyncio.to_thread(manager.get(episode_id).cancel,action_id)

    @app.get('/episodes/{episode_id}/images/{revision}/{camera_id}')
    def image(episode_id:str,revision:int,camera_id:str,request:Request):
        # FastAPI runs sync routes in its thread pool. Native RPCs hold this
        # lock while updating a decision snapshot, so never take it on ASGI's
        # event loop: health and cancellation requests must remain schedulable.
        session=manager.media_session(episode_id,bearer(request))
        with session.lock:
            path=session.public_images.get((revision,camera_id))
        if path is None:
            raise ServiceError(404,'Unknown public camera frame')
        return FileResponse(path,media_type='image/png')

    @app.get('/episodes/{episode_id}/depth/{revision}/{camera_id}')
    async def depth(episode_id:str,revision:int,camera_id:str,request:Request):
        value=await asyncio.to_thread(manager.media_session(episode_id,bearer(request)).read_public_depth,revision,camera_id)
        return Response(value,media_type='application/octet-stream')

    @app.post('/episodes/{episode_id}/finish')
    async def finish(episode_id:str):
        return await asyncio.to_thread(manager.get(episode_id).finish)

    @app.get('/episodes/{episode_id}/result')
    def result(episode_id:str):
        # Serialize the same committed result under the session lock, off-loop.
        session=manager.get(episode_id)
        with session.lock:
            if session.result is None:
                raise ServiceError(409,'Episode is still running')
            return session.result.model_dump(mode='json')

    return app


app=create_app()
