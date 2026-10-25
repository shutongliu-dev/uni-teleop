"""Compact, function-specific rooms; all dimensions are clear interior metres."""

# Authored lighting choices, not measured illuminance. Keep task fixtures, room
# fixtures and daylight independently addressable for observation validation.
FAMILY_LIGHTING = {
    'mechanical_assembly': dict(temperature_k=4600., ceiling_scale=1.10, sun=1050., sky=410.),
    'laboratory': dict(temperature_k=5000., ceiling_scale=1.50, sun=1100., sky=460.),
    'kitchen': dict(temperature_k=4000., ceiling_scale=1.00, sun=1600., sky=410.),
    'study': dict(temperature_k=3300., ceiling_scale=1.25, sun=1550., sky=420.),
    'card_room': dict(temperature_k=3000., ceiling_scale=1.05, sun=1000., sky=340.),
    'living_room': dict(temperature_k=3500., ceiling_scale=1.30, sun=1450., sky=450.),
}

LIGHTING_PROFILES = {
    'nominal': dict(scales=dict(ceiling=1., task=1., accent=1., sun=1., sky=1.), temperature_shift_k=0.),
    'daylight': dict(scales=dict(ceiling=.70, task=.85, accent=.50, sun=1.65, sky=1.40), temperature_shift_k=0.),
    'warm': dict(scales=dict(ceiling=.70, task=.90, accent=1.10, sun=.15, sky=.40), temperature_shift_k=-500.),
    'low_light': dict(scales=dict(ceiling=.22, task=.38, accent=.45, sun=.03, sky=.12), temperature_shift_k=0.),
}

ROOM_SPECS = {
    'mechanical_assembly': dict(bounds=(-1.45, 2.65, -1.60, 2.05), height=2.80,
        theme='compact maintenance workshop', window=dict(wall='left', center=.35, width=1.35, bottom=1.62, top=2.38),
        camera=dict(eye=[2.30,-1.35,2.18],target=[.85,.75,1.10],focal_length_mm=23)),
    'laboratory': dict(bounds=(-1.60, 2.65, -1.55, 2.05), height=2.65,
        theme='clean sample preparation laboratory', window=dict(wall='left', center=.30, width=1.45, bottom=1.30, top=2.23),
        camera=dict(eye=[2.30,-1.30,2.10],target=[.85,.75,1.10],focal_length_mm=23)),
    'kitchen': dict(bounds=(-1.65, 2.65, -1.65, 2.05), height=2.65,
        theme='compact domestic L-shaped kitchen', window=dict(wall='back', center=-.83, width=1.22, bottom=1.12, top=2.22),
        camera=dict(eye=[2.29,-1.40,2.08],target=[.85,.75,1.10],focal_length_mm=23)),
    'study': dict(bounds=(-1.45, 2.50, -1.65, 1.90), height=2.65,
        theme='warm built-in library and writing study', window=dict(wall='left', center=.25, width=1.25, bottom=.82, top=2.28),
        camera=dict(eye=[2.18,-1.40,2.08],target=[.85,.75,1.10],focal_length_mm=23)),
    'card_room': dict(bounds=(-1.45, 2.55, -1.70, 1.95), height=2.70,
        theme='Chinese tea and Go room', window=dict(wall='back', center=.35, width=1.58, bottom=1.06, top=2.28),
        camera=dict(eye=[2.20,-1.44,2.06],target=[.30,.38,1.0],focal_length_mm=23)),
    'living_room': dict(bounds=(-1.55, 2.85, -1.80, 2.10), height=2.70,
        theme='domestic lounge with media wall', window=dict(wall='left', center=.48, width=2.35, bottom=.25, top=2.40),
        camera=dict(eye=[-1.30,-1.55,2.10],target=[.60,.75,1.05],focal_length_mm=23)),
}


def cameras_for(family):
    s=ROOM_SPECS[family]
    return {
        'room':s['camera'],
        'workcell':dict(eye=[2.08,-1.36,1.75],target=[.35,.12,1.02],focal_length_mm=30),
        'tabletop':dict(eye=[1.60,-1.32,1.68],target=[.65,.0,.84],focal_length_mm=41),
    }
