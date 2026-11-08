"""Public card-game adapters; no scene, reset or simulator access."""
from copy import deepcopy
import json

from bench.protocol.card_games import INTENT_ADAPTER
from .bench_http import BenchHTTPError


CARD_PROTOCOL = 'bench_card_game_state_v1'
CARD_TASKS = frozenset(('polyomino_territory', 'barrier_race', 'memory_pairs', 'disk_curling'))
CARD_INTENTS = {
    'polyomino_territory': ('place_piece', 'pass', 'cancel_intent'),
    'barrier_race': ('move_pawn', 'place_barrier', 'cancel_intent'),
    'memory_pairs': ('reveal_card', 'collect_pair', 'flip_back', 'cancel_intent'),
    'disk_curling': ('shoot_disk', 'cancel_intent'),
}
BASELINE_GUIDANCE = (
    'A fresh card-game reset has not advanced native physics. Before the first intent, '
    'the current player should explicitly hold(duration_s=0.5, mode="acknowledged_drive") '
    'to let the referee sample real baseline/support evidence, then inspect state again. '
    'Read-only queries do not advance physics. If the referee is still awaiting baseline '
    'evidence, use another explicit hold and inspect feedback; a missing baseline is not a completed game.')
_EXAMPLES = {
    'polyomino_territory': [{'type': 'place_piece', 'piece_id': 'black_tet_L', 'anchor': [0, 0], 'rotation_deg': 90},
                          {'type': 'pass'}],
    'barrier_race': [{'type': 'move_pawn', 'destination': [0, 1]},
                     {'type': 'place_barrier', 'piece_id': 'black_barrier_0', 'anchor': [0, 0], 'orientation': 'horizontal'}],
    'memory_pairs': [{'type': 'reveal_card', 'card_id': 'card_00'}, {'type': 'collect_pair'}, {'type': 'flip_back'}],
    'disk_curling': [{'type': 'shoot_disk', 'piece_id': 'black_disk_0'}],
}


def intent_tools(task_id, color='black'):
    """Describe task-specific declarations; motion always uses robot functions."""
    allowed = CARD_INTENTS[task_id]
    schema = deepcopy(INTENT_ADAPTER.json_schema())
    schema['oneOf'] = [row for row in schema['oneOf'] if
        schema['$defs'][row['$ref'].rsplit('/', 1)[-1]]['properties']['type']['const'] in allowed]
    definitions = {row['$ref'].rsplit('/', 1)[-1] for row in schema['oneOf']}
    schema['$defs'] = {key: value for key, value in schema['$defs'].items() if key in definitions}
    if 'discriminator' in schema:
        schema['discriminator']['mapping'] = {key: value for key, value in
            schema['discriminator']['mapping'].items() if key in allowed}
    examples = [{'intent': item} for item in deepcopy(_EXAMPLES[task_id]) + [{'type': 'cancel_intent'}]]
    for row in examples:
        if 'piece_id' in row['intent']:
            row['intent']['piece_id'] = color + row['intent']['piece_id'][5:]
    scope = ('Declare only an operation permitted by board_game.phase and public_rules. '
             'The intent does not move an object. Use the existing owned robot functions, release, '
             'and wait for physical referee settlement. turn_index is an operation authorization epoch; '
             'the same player may remain current after it changes. Never infer a player from its parity. '
             'Example IDs and coordinates illustrate syntax; check the public phase and images before selecting them.')
    scope += ' ' + BASELINE_GUIDANCE
    if task_id == 'memory_pairs':
        scope += (' Reveal one card at a time; every real public reveal, including an unexpected one, '
                  'is delivered to both players in public_game_events history. Cancellation cannot erase '
                  'a reveal. Collect a matched pair or flip a mismatch back using real robot motion; '
                  'never assume a hidden pair identity.')
    if task_id == 'disk_curling':
        scope += (' Push only the selected unplayed disk from your launch region and release it. '
                  'The referee settles all played disks after free motion; a push response is not a score.')
    return {'intent': schema, 'scope': scope}, examples


class PublicEventDelivery:
    """Drain each authenticated player's log into that player's own text history.

    A historical reveal survives a later face-down state and a long opponent turn.
    Events are compared across players; opponent reasoning is never copied.
    """
    def __init__(self, clients):
        self.clients = clients
        self.cursors = dict.fromkeys(clients, -1)
        self.events = {key: [] for key in clients}
        self._canonical = {}

    def drain(self, histories, *, through=None):
        batches = {}
        for player, client in self.clients.items():
            collected = []
            for _ in range(4096):
                cursor = self.cursors[player]
                page = client.events(after=cursor, limit=256)
                rows = page['events']
                expected = list(range(cursor + 1, cursor + 1 + len(rows)))
                if ([row['sequence'] for row in rows] != expected
                        or page['next_cursor'] != cursor + len(rows)
                        or (page['has_more'] and not rows)):
                    raise BenchHTTPError('Invalid public game event sequence', category='invalid_response')
                for row in rows:
                    encoded = json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)
                    previous = self._canonical.setdefault(row['sequence'], encoded)
                    if previous != encoded:
                        raise BenchHTTPError('Players received different public game events', category='invalid_response')
                # Persist each page before requesting the next page, including on failure.
                if rows:
                    histories[player].append({'role': 'user', 'content': json.dumps(
                        {'public_game_events': rows}, ensure_ascii=False, allow_nan=False)})
                    self.events[player].extend(deepcopy(rows))
                    collected.extend(rows)
                self.cursors[player] = page['next_cursor']
                if not page['has_more']:
                    break
            else:
                raise BenchHTTPError('Public game event pagination did not terminate', category='invalid_response')
            if through is not None and self.cursors[player] < through:
                raise BenchHTTPError('Public game events omit observed events', category='invalid_response')
            batches[player] = collected
        # No physics runs while the host is draining an idle operation epoch.
        if len(set(self.cursors.values())) > 1:
            raise BenchHTTPError('Players received incomplete public game events', category='invalid_response')
        return batches


def committed_operation_counts(delivery):
    """Count verified public physical commits once; pass/reveal notices give no credit.

    A delivery has already checked both authenticated logs for identity and
    completeness. Count the common log once, rather than once per recipient.
    This is an operator validation target, never an alternative game scorer.
    """
    counts = dict.fromkeys(delivery.clients, 0)
    rows = next(iter(delivery.events.values()), [])
    for row in rows:
        if row['type'] == 'operation_committed' and row['details']['operation'] != 'pass':
            counts[row['player_id']] += 1
    return counts
