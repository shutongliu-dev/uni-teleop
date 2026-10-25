"""Pinned CC0 scanned food and cookware, composed as static kitchen scenery."""
import math

from bench.paths import load_json


def build_dressing(b):
    root=b.assets_root/'objects/polyhaven/derived'
    placements=[]

    def asset(name,slug,xyz,yaw=0,collection='kitchen_v1',extension='usdc'):
        directory=root/collection/slug
        path=b.asset('kitchen_dressing_'+name,directory/(slug+'.'+extension),xyz,
                     rotation=(0,0,yaw),ground=True)
        metadata=directory/'DERIVATION.json'; b.dependency(metadata)
        record=load_json(metadata)
        for key in ('source_derivation','source_collection'):
            if record.get(key):
                b.dependency(record[key])
        if record.get('source_provenance'):
            b.dependency(record['source_provenance'])
            from pathlib import Path
            b.dependency(Path(record['source_provenance']).with_name('LICENSE.txt'))
        placements.append({'name':name,'asset_id':slug,'collection':collection,'path':path,'support_z_m':xyz[2],
                           'placement_m':list(xyz),'scale':1.,'static_scenery':True})

    # Prep station at the left end of the bar; the central manipulation area
    # remains clear. Original scanned scale and base alignment are preserved.
    asset('cutting_board','wooden_cutting_board',(-.48,-.78,.90),-9)
    asset('apple','food_apple_01',(-.58,-.77,.9371388),15)
    asset('onion','yellow_onion',(-.40,-.76,.9384488),-20)
    asset('wooden_spoon','wooden_spoon',(-.83,-.85,.90),-15)
    asset('fruit_bowl','wooden_bowl_01',(-.44,-1.17,.90),collection='kitchen_v2')
    # Bowl contents sit inside the scanned cavity, above its interior floor.
    asset('bowl_apple','food_apple_01',(-.48,-1.19,.921),-35)
    asset('bowl_lemon','lemon',(-.395,-1.180,.9258),65)
    asset('bowl_onion','yellow_onion',(-.433,-1.117,.92756),10)
    # Breakfast plate with a physically hollow profile, a real cup and handle.
    b.material('kitchen_dressing_glaze',(.70,.755,.69),.20)
    b.lathe('kitchen_breakfast_plate',(1.57,-1.06,.90),
            [(.0,0),(.095,0),(.13,.012),(.133,.020),(.125,.023),(.094,.009),(.0,.009)],
            'kitchen_dressing_glaze',collision=True)
    asset('croissant','croissant',(1.57,-1.06,.910),-12)
    b.lathe('kitchen_breakfast_mug',(1.34,-.71,.90),
            [(.028,0),(.039,.005),(.043,.09),(.038,.093),(.034,.013),(.0,.013),(.0,0)],
            'ceramic',collision=True)
    points=[(1.382+.028*math.sin(t),-.71,.948+.032*math.cos(t))
            for t in [i*math.pi/24 for i in range(25)]]
    b.tube('kitchen_breakfast_mug_handle',points,.006,'ceramic',collision=True)
    b.cylinder('kitchen_coffee_surface',(1.34,-.71,.977),.036,.0015,'walnut',collision=False)
    asset('vintage_vase','metal_jug',(1.68,-.71,.90),25,collection='kitchen_v2')
    b.material('kitchen_dried_wheat',(.58,.40,.17),.86)
    # The weathered upstream milk can is reused as a dry wheat vase.
    for i,(dx,dy,height) in enumerate(((-.055,-.025,.52),(.04,.016,.57),
                                      (.015,-.034,.48),(-.018,.04,.55),(.065,.035,.46))):
        top=(1.68+dx,-.71+dy,.90+height)
        b.tube('kitchen_wheat_stem_'+str(i),[(1.68,-.71,1.03),
               (1.68+dx*.65,-.71+dy*.65,1.25),top],.0017,'kitchen_dried_wheat')
        for row in range(7):
            z=top[2]-.075+row*.01
            for sign in (-1,1):
                b.sphere(f'kitchen_wheat_grain_{i}_{row}_{sign}',
                         (top[0]+sign*.004,top[1],z),(.004,.003,.008),
                         'kitchen_dried_wheat')
    b.box('kitchen_bread_serving_board',(1.50,-1.27,.904),(.48,.14,.008),
          'kitchen_arch_oak',.003,collision=True)
    asset('bread_rolls','hamburger_buns',(1.50,-1.27,.908),collection='kitchen_v2')
    # Pantry goods occupy the outside of the right-hand cooking run.
    asset('pantry','long_life_food',(1.52,1.15,.90))
    asset('lemon','lemon',(1.45,.83,.90),65)
    asset('enamel_pot','pot_enamel_01',(.815,1.095,.901),-8)
    asset('banana_bunch','banana_bunch',(-.13,1.08,.90),-15,
          collection='kitchen_v2_selected',extension='usda')
    # The native source includes a long handle. Its bounding-box centre is
    # offset from the pan centre by 127.354 mm; yaw -90 points the handle +X.
    asset('brass_pan','brass_pan_01',(.942354,.815,.901),-90,collection='kitchen_v2')
    for collection in ('kitchen_v1','kitchen_v2'):
        b.dependency(root/collection/'COLLECTION.json')
    return {'schema':'kitchen_static_dressing.v2','assets':placements,
            'original_scale_preserved':True,'source':'Poly Haven CC0, kitchen_v1, kitchen_v2 and explicit banana selection',
            'scope':'Static set dressing; no movable-object physics or task qualification'}
