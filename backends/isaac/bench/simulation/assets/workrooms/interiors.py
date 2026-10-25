"""Compact, function-specific perimeter furniture; shared workcell stays untouched."""
import math
from .layouts import ROOM_SPECS
from .common import light


class _Frame:
    """Author joinery in an XY frame, including complete referenced mesh props."""
    def __init__(self,b,origin=(0,0,0),yaw=0):
        self.b=b;self.origin=origin;self.yaw=yaw
        a=math.radians(yaw);self.c=math.cos(a);self.s=math.sin(a)
    def __getattr__(self,name):return getattr(self.b,name)
    def point(self,p):
        x,y,z=p;ox,oy,oz=self.origin
        return ox+self.c*x-self.s*y,oy+self.s*x+self.c*y,oz+z
    def rotation(self,r):return r[0],r[1],r[2]+self.yaw
    def box(self,n,p,size,mat,bevel=.002,rotation=(0,0,0),collision=True):
        return self.b.box(n,self.point(p),size,mat,bevel,rotation=self.rotation(rotation),collision=collision)
    def cylinder(self,n,p,r,h,mat,rotation=(0,0,0),collision=True):
        return self.b.cylinder(n,self.point(p),r,h,mat,rotation=self.rotation(rotation),collision=collision)
    def sphere(self,n,p,scale,mat,collision=False):return self.b.sphere(n,self.point(p),scale,mat,collision=collision)
    def lathe(self,n,p,profile,mat,rotation=(0,0,0),collision=False):
        return self.b.lathe(n,self.point(p),profile,mat,rotation=self.rotation(rotation),collision=collision)
    def tube(self,n,points,r,mat,collision=False):return self.b.tube(n,[self.point(p) for p in points],r,mat,collision=collision)
    def label(self,n,text,p,size,material='paper',rotation=(0,0,0),font_size=80):
        return self.b.label(n,text,self.point(p),size,material,rotation=self.rotation(rotation),font_size=font_size)
    def asset(self,n,path,p,scale=1.,rotation=(0,0,0),ground=True):
        return self.b.asset(n,path,self.point(p),scale=scale,rotation=self.rotation(rotation),ground=ground)


def _ph(b,name,slug,xyz,yaw=0.,scale=1.,version='v1'):
    return b.asset(name,b.assets_root/'objects/polyhaven/derived'/version/slug/(slug+'.usdc'),xyz,scale=scale,rotation=(0,0,yaw))


def _sim(b,name,relative,xyz,yaw=0.,scale=1.):
    return b.asset(name,b.assets_root/'objects/simready/v1/Industrial'/relative,xyz,scale=scale,rotation=(0,0,yaw))


def _materials(b):
    for n,c,r,m in [
        ('linen_ivory',(.68,.65,.57),.94,0),('rug_border',(.22,.25,.26),.95,0),
        ('sage_enamel',(.28,.37,.32),.41,0),('tile_cream',(.74,.76,.71),.22,0),
        ('grout',(.37,.40,.37),.9,0),('quartz',(.70,.70,.66),.29,0),
        ('lab_gray',(.56,.62,.63),.42,0),('lab_teal',(.12,.30,.33),.45,0),
        ('warm_cream',(.83,.78,.67),.68,0),('workshop_blue',(.085,.17,.24),.39,.3),
        ('library_ochre',(.37,.21,.09),.7,0),('book_burgundy',(.21,.045,.045),.72,0),
        ('screen_black',(.012,.018,.021),.16,.15),('tea_porcelain',(.58,.68,.59),.19,0)]:
        b.material(n,c,r,m)
    b.material('hood_glazing',(.68,.82,.83),.1,opacity=.12)


def _rug(b,n,xy,size):
    x,y=xy;w,d=size
    b.box(n+'_edge',(x,y,.006),(w,d,.012),'rug_border',.006,collision=False)
    b.box(n+'_pile',(x,y,.013),(w-.055,d-.055,.008),'linen_ivory',.004,collision=False)
    for s in (-1,1):
        b.box(n+'_seam_x'+str(s),(x+s*(w/2-.045),y,.0172),(.002,d-.09,.0008),'fabric',0,collision=False)
        b.box(n+'_seam_y'+str(s),(x,y+s*(d/2-.045),.0172),(w-.09,.002,.0008),'fabric',0,collision=False)


def _cabinet(b,n,x,width=.59,height=.84,depth=.57,mat='white',drawers=False):
    """Furniture front is -Y; casework top is exactly height."""
    # Hollow carcass leaves the sink bowl and closed drawers physical interior space.
    b.box(n+'_back',(x,depth/2-.012,(height+.09)/2),(width,.024,height-.09),'white',.003)
    b.box(n+'_bottom',(x,0,.102),(width,depth,.024),'white',.003)
    for side in (-1,1):b.box(n+'_side'+str(side),(x+side*(width/2-.012),0,(height+.09)/2),(.024,depth,height-.09),'white',.003)
    b.box(n+'_plinth',(x,.035,.045),(width-.05,depth-.11,.09),'black',.002)
    if drawers:
        for j in range(3):
            h=(height-.12)/3;z=.105+h*(j+.5)
            b.box(n+'_drawer'+str(j),(x,-depth/2-.014,z),(width-.014,.025,h-.009),mat,.004)
            b.tube(n+'_pull'+str(j),[(x-.10,-depth/2-.029,z+.05),(x-.10,-depth/2-.055,z+.05),(x+.10,-depth/2-.055,z+.05),(x+.10,-depth/2-.029,z+.05)],.006,'chrome')
    else:
        for s in (-1,1):
            xx=x+s*width/4
            b.box(n+'_door'+str(s),(xx,-depth/2-.014,(height+.09)/2),(width/2-.009,.025,height-.105),mat,.004)
            b.box(n+'_pull'+str(s),(x+s*.035,-depth/2-.047,height*.73),(.009,.025,.13),'chrome',.003,collision=False)


def _books(b,n,x,y,z,count=14,offset=0):
    mats=['paper','navy','book_burgundy','library_ochre','green','linen_ivory']
    for i in range(count):
        w=.027+.006*((i+offset)%4);h=.18+.021*((i*3+offset)%5)
        b.box(n+str(i),(x+w/2,y,z+h/2),(w,.16,h),mats[(i+offset)%6],.0015,collision=False)
        for dz in (.038,h-.040):b.box(n+str(i)+'_band'+str(dz),(x+w/2,y-.081,z+dz),(w*.72,.001,.003),'paper',0,collision=False)
        x+=w+.004


def _mechanical(b):
    rear=ROOM_SPECS['mechanical_assembly']['bounds'][3]
    f=_Frame(b,(0,rear-.335,0))
    for i,x in enumerate([-.92,-.30,.32,.94]):_cabinet(f,'service_base'+str(i),x,.605,.86,.58,'workshop_blue',drawers=i in (0,3))
    f.box('service_oak_top',(0,0,.888),(2.49,.65,.056),'oak',.008)
    f.box('service_back_lip',(0,.306,.94),(2.49,.025,.10),'steel',.003)
    py=rear-.050
    b.box('workshop_pegboard',(0,py,1.63),(2.46,.035,1.12),'workshop_blue',.005)
    for ix in range(31):
        for iz in range(13):b.cylinder('peg_'+str(ix)+'_'+str(iz),(-1.16+ix*.077,py-.020,1.10+iz*.081),.004,.002,'black',rotation=(90,0,0),collision=False)
    for i,x in enumerate([-.99,-.65,-.31,.03,.37]):
        b.tube('service_hook'+str(i),[(x,py-.024,1.82),(x,py-.10,1.82),(x,py-.10,1.85)],.004,'chrome')
        b.box('service_spanner_body'+str(i),(x,py-.09,1.67),(.026,.013,.29),'steel',.006,collision=False)
        b.cylinder('service_spanner_ring'+str(i),(x,py-.09,1.835),.029,.014,'steel',rotation=(90,0,0),collision=False)
        b.cylinder('service_spanner_hole'+str(i),(x,py-.100,1.835),.015,.001,'black',rotation=(90,0,0),collision=False)
    b.box('peg_small_parts_shelf',(.79,py-.10,1.44),(.60,.20,.023),'steel',.003)
    for i in range(4):
        x=.56+i*.145
        b.box('peg_bin'+str(i),(x,py-.11,1.5015),(.125,.16,.10),'navy',.008)
        b.label('peg_bin_tag'+str(i),'M'+str(4+i*2),(x,py-.193,1.51),(.075,.035,.001),rotation=(90,0,0),font_size=32)
    _sim(f,'service_toolbox','Tools/Hand/Large_Tool_Box_A01/obs_electricians_large_tool_box_01.usd',(-.78,0,.916),scale=.85)
    _sim(f,'service_hammer','Tools/Hand/Cross_Peen_Hammer_B01/sm_handTool_hammer_crossPeen_b01_01.usd',(.18,-.02,.916))
    _sim(f,'service_lubricant','Hardware/Lubricants/WD40_Silicone_Lubricant_A01/sm_consumable_lubricant_silicone_wd40_a01_01.usd',(.84,.04,.916))
    _sim(b,'assembly_shelf','Warehouse/Racks/Heavy_Duty_Steel_Shelve_A06/sm_shelve_steel_heavyDuty_a06_01.usd',(1.97,rear-.34,0))
    _sim(b,'assembly_packing_box','Warehouse/Boxes/Cardboard_Open_Box_A01/sm_box_cardboard_open_a01_01.usd',(1.97,rear-.34,.08336),scale=.85)
    b.box('shelf_spares_case',(1.97,rear-.34,1.35475),(.65,.35,.206),'navy',.012)
    b.box('shelf_spares_handle',(1.97,rear-.525,1.365),(.18,.025,.045),'black',.005,collision=False)
    b.tube('workshop_conduit',[(-1.34,rear-.025,.14),(-1.34,rear-.025,2.40),(1.24,rear-.025,2.40)],.013,'steel')
    b.box('workshop_socket',(-1.31,rear-.055,1.05),(.13,.055,.14),'white',.006)


def _laboratory(b):
    rear=ROOM_SPECS['laboratory']['bounds'][3];f=_Frame(b,(0,rear-.335,0))
    for i,x in enumerate([-1.08,-.43,.22,.87]):_cabinet(f,'clean_casework'+str(i),x,.635,.84,.58,'lab_gray',drawers=True)
    f.box('lab_epoxy_counter',(-.105,0,.865),(2.62,.65,.05),'black',.007)
    h=_Frame(b,(-.76,rear-.335,0))
    h.box('hood_back',(0,.291,1.50),(1.28,.035,1.22),'white',.005)
    for s in (-1,1):h.box('hood_side'+str(s),(s*.625,0,1.50),(.045,.60,1.22),'white',.004)
    h.box('hood_header',(0,0,2.09),(1.30,.60,.16),'white',.005)
    h.box('hood_sash_top',(0,-.31,1.95),(1.21,.04,.035),'steel',.002)
    h.box('hood_sash_bottom',(0,-.31,1.63),(1.21,.04,.035),'steel',.002)
    for s in (-1,1):h.box('hood_sash_side'+str(s),(s*.59,-.31,1.79),(.026,.04,.35),'steel',.002)
    h.box('hood_glass',(0,-.305,1.79),(1.16,.007,.29),'hood_glazing',0,collision=False)
    for i in range(8):h.box('hood_baffle'+str(i),(0,.270,1.04+i*.068),(.91,.002,.008),'black',.001,collision=False)
    h.label('hood_title','SAMPLE PREPARATION',(0,-.307,2.09),(1.04,.065,.001),rotation=(90,0,0),font_size=40)
    for i,x in enumerate([-.43,-.18,.07,.32]):
        h.lathe('hood_reagent'+str(i),(x,.06,.89),[(.029,0),(.029,.10),(.018,.13),(.018,.155),(0,.155)],'ceramic')
        h.cylinder('hood_reagent_cap'+str(i),(x,.06,1.052),.019,.014,'lab_teal',collision=False)
    _ph(f,'laboratory_microscope','industrial_microscope',(.78,-.03,.89),version='v2',scale=.90)
    s=_Frame(b,(1.86,rear-.31,0));w=.94
    s.box('clean_storage_toe',(0,.02,.055),(w-.05,.49,.11),'black',.003)
    s.box('clean_storage_back',(0,.275,1.19),(w,.025,2.16),'white',.004)
    for side in (-1,1):s.box('clean_storage_side'+str(side),(side*(w/2-.015),0,1.19),(.03,.57,2.16),'white',.004)
    for i,z in enumerate([.11,.78,1.20,1.62,2.04,2.27]):s.box('clean_storage_shelf'+str(i),(0,0,z),(w,.57,.025),'white',.003)
    for side in (-1,1):
        x=side*w/4
        s.box('clean_storage_lower_door'+str(side),(x,-.300,.445),(w/2-.011,.025,.635),'lab_teal',.004)
        s.box('clean_storage_handle'+str(side),(side*.035,-.330,1.30),(.01,.025,.20),'chrome',.003)
        s.box('clean_storage_glass'+str(side),(x,-.296,1.53),(w/2-.064,.006,1.41),'hood_glazing',0,collision=False)
        for dx in (-1,1):s.box('clean_storage_stile'+str(side)+'_'+str(dx),(x+dx*(w/4-.016),-.300,1.53),(.026,.025,1.49),'white',.002)
        for z in (.792,2.267):s.box('clean_storage_rail'+str(side)+'_'+str(z),(x,-.300,z),(w/2-.012,.025,.027),'white',.002)
    for row,z in enumerate([.793,1.213,1.633]):
        for i,x in enumerate([-.28,0,.28]):
            s.box('sterile_box'+str(row)+'_'+str(i),(x,.025,z+.075),(.22,.32,.15),'linen_ivory',.004)
            s.label('sterile_tag'+str(row)+'_'+str(i),'STERILE',(x,-.138,z+.080),(.15,.04,.001),rotation=(90,0,0),font_size=32)
    f=_Frame(b,(-1.30,.32,0),90)
    _cabinet(f,'lab_wash_base',0,1.05,.82,.52,'white')
    for n,p,s in [('front',(0,-.235,.845),(1.08,.09,.05)),('back',(0,.235,.845),(1.08,.09,.05)),('left',(-.395,0,.845),(.29,.38,.05)),('right',(.395,0,.845),(.29,.38,.05))]:
        f.box('lab_wash_top_'+n,p,s,'lab_gray',.005)
    f.box('lab_wash_basin_floor',(0,0,.725),(.49,.37,.012),'steel',.012)
    for side in (-1,1):
        f.box('lab_wash_basin_x'+str(side),(side*.242,0,.797),(.012,.37,.134),'steel',.003)
        f.box('lab_wash_basin_y'+str(side),(0,side*.179,.797),(.49,.012,.134),'steel',.003)
    f.cylinder('lab_wash_drain',(0,0,.732),.023,.002,'black',collision=False)
    f.tube('lab_wash_tap',[(0,.20,.87),(0,.20,1.07),(0,.13,1.10),(0,.03,1.10),(0,.0,1.06)],.012,'chrome')


def _kitchen(b):
    rear=ROOM_SPECS['kitchen']['bounds'][3];f=_Frame(b,(0,rear-.335,0))
    _cabinet(f,'kitchen_sink_base',-.94,1.188,.84,.58,'sage_enamel')
    for i,x in enumerate([-.04,1.28]):_cabinet(f,'kitchen_base'+str(i),x,.588,.84,.58,'sage_enamel')
    for n,p,s in [('front',(-.64,-.265,.8625),(1.82,.12,.045)),('back',(-.64,.2675,.8625),(1.82,.115,.045)),('left',(-1.32,0,.8625),(.46,.415,.045)),('right',(-.155,0,.8625),(.85,.415,.045)),('right_end',(1.28,0,.8625),(.64,.65,.045))]:
        f.box('kitchen_counter_'+n,p,s,'quartz',.004)
    f.box('kitchen_sink_floor',(-.83,0,.716),(.49,.38,.014),'steel',.018)
    for side in (-1,1):
        f.box('kitchen_sink_x'+str(side),(-.83+side*.244,0,.79),(.012,.39,.14),'steel',.003)
        f.box('kitchen_sink_y'+str(side),(-.83,side*.188,.79),(.50,.014,.14),'steel',.003)
    f.cylinder('kitchen_sink_drain',(-.83,0,.725),.024,.002,'black',collision=False)
    f.tube('kitchen_tap',[(-.83,.25,.885),(-.83,.25,1.09),(-.83,.20,1.145),(-.83,.08,1.145),(-.83,.015,1.09)],.013,'chrome')
    f.box('stove_leveling_plinth',(.64,0,.013),(.485,.61,.026),'black',.002)
    _ph(f,'kitchen_real_stove','electric_stove',(.64,0,.026))
    f.box('extractor_hood',(.64,.02,1.64),(.60,.49,.10),'steel',.01)
    f.box('extractor_chimney',(.64,.14,2.04),(.25,.24,.70),'steel',.004)
    for i in range(9):f.box('extractor_slot'+str(i),(.44+i*.05,-.226,1.64),(.018,.002,.04),'black',.001,collision=False)
    f.box('kitchen_upper',(1.28,.13,1.98),(.59,.35,.75),'white',.005)
    f.box('kitchen_upper_door',(1.28,-.059,1.98),(.578,.024,.738),'warm_cream',.004)
    f.box('kitchen_upper_pull',(1.49,-.08,1.74),(.01,.025,.14),'chrome',.003)
    side=_Frame(b,(-1.315,.66,0),90)
    for i,x in enumerate([-.38,.38]):_cabinet(side,'kitchen_return'+str(i),x,.75,.84,.56,'sage_enamel',drawers=True)
    side.box('kitchen_return_counter',(0,0,.8625),(1.54,.62,.045),'quartz',.004)
    for k,z in enumerate([1.42,1.80]):
        side.box('kitchen_spice_shelf'+str(k),(-.10,.22,z),(1.22,.23,.026),'oak',.004)
        for x in (-.55,.35):side.box('spice_bracket'+str(k)+'_'+str(x),(x,.3175,z-.09),(.028,.035,.16),'black',.002)
        for i in range(4):
            x=-.55+i*.24
            side.lathe('spice_jar'+str(k)+'_'+str(i),(x,.20,z+.013),[(.033,0),(.033,.09),(.030,.11),(0,.11)],'ceramic')
            side.cylinder('spice_lid'+str(k)+'_'+str(i),(x,.20,z+.129),.033,.012,'walnut',collision=False)
    f.box('fridge_body',(2.12,-.025,.99),(.77,.65,1.98),'white',.017)
    for n,z,h in [('upper',1.24,1.42),('freezer',.27,.49)]:
        f.box('fridge_gasket_'+n,(2.12,-.359,z),(.735,.018,h),'black',.005)
        f.box('fridge_door_'+n,(2.12,-.378,z),(.725,.028,h-.015),'white',.014)
        f.box('fridge_grip_'+n,(1.82,-.402,z),(.014,.023,min(h*.55,.45)),'steel',.004)
    for i in range(9):f.box('fridge_vent'+str(i),(1.84+i*.07,-.365,.052),(.035,.003,.018),'black',.001,collision=False)


def _study(b):
    rear=ROOM_SPECS['study']['bounds'][3];y=rear-.225
    for bay,x in enumerate([-.72,.48,1.68]):
        n='library_bay'+str(bay)
        b.box(n+'_toe',(x,y+.018,.055),(1.17,.34,.11),'walnut',.003)
        b.box(n+'_back',(x,rear-.025,1.24),(1.19,.025,2.38),'walnut',.003)
        for s in (-1,1):b.box(n+'_side'+str(s),(x+s*.585,y,1.24),(.03,.39,2.38),'oak',.004)
        for j,z in enumerate([.11,.68,1.07,1.46,1.85,2.24,2.43]):
            b.box(n+'_shelf'+str(j),(x,y,z),(1.20,.39,.03),'oak',.004)
            if 1<=j<=4:
                count=18 if (j+bay)%2 else 13
                _books(b,n+'_books'+str(j)+'_',x-.545,y-.015,z+.015,count,offset=bay+j)
                if count==13:
                    for k in range(3):b.box(n+'_stack'+str(j)+'_'+str(k),(x+.37,y-.015,z+.03+k*.031),(.27,.17,.028),['navy','paper','book_burgundy'][k],.002,collision=False)
        for s in (-1,1):
            b.box(n+'_cupboard'+str(s),(x+s*.297,y-.211,.40),(.581,.028,.532),'oak',.004)
            b.cylinder(n+'_knob'+str(s),(x+s*.045,y-.235,.45),.013,.018,'chrome',rotation=(90,0,0),collision=False)
    b.box('library_crown',(.48,y,2.465),(3.66,.43,.055),'walnut',.005)
    f=_Frame(b,(-1.17,.20,0),90)
    f.box('reading_bench_toe',(0,.02,.055),(1.22,.43,.11),'walnut',.003)
    f.box('reading_bench_base',(0,0,.235),(1.28,.49,.36),'oak',.006)
    for i,x in enumerate([-.32,.32]):
        f.box('reading_bench_drawer'+str(i),(x,-.255,.25),(.62,.024,.29),'oak',.004)
        f.box('reading_bench_pull'+str(i),(x,-.278,.29),(.13,.02,.012),'chrome',.003)
    f.box('reading_bench_cushion',(0,0,.465),(1.25,.48,.10),'fabric',.04,collision=False)
    f.box('reading_bench_pillow',(-.42,.06,.57),(.30,.28,.13),'linen_ivory',.055,rotation=(0,12,8),collision=False)


def _card(b):
    rear=ROOM_SPECS['card_room']['bounds'][3]
    _ph(b,'tea_chinese_commode','chinese_commode',(.35,rear-.31,0),version='v2',scale=.42)
    _ph(b,'tea_left_chair','chinese_armchair',(-1.05,-.78,0),yaw=65,version='v2',scale=.66)
    _ph(b,'tea_right_chair','chinese_armchair',(1.92,.86,0),yaw=-18,version='v2',scale=.72)
    f=_Frame(b,(-1.417,.62,0),90)
    f.box('tea_scroll_back',(0,0,1.76),(.66,.012,.92),'linen_ivory',.002,collision=False)
    for z in (1.29,2.23):f.cylinder('tea_scroll_rod'+str(z),(0,-.008,z),.018,.74,'walnut',rotation=(0,90,0),collision=False)
    f.label('tea_calligraphy','静  观\n观棋不语',(0,-.009,1.77),(.53,.74,.001),rotation=(90,0,0),font_size=83)


def _living(b):
    rear=ROOM_SPECS['living_room']['bounds'][3]
    _ph(b,'living_sofa','sofa_02',(-.30,rear-.46,.017))
    _ph(b,'living_armchair','modern_arm_chair_01',(1.52,rear-.52,.017),scale=.92)
    _rug(b,'living_seating_rug',(.30,1.57),(3.35,.98))
    _ph(b,'living_console','modern_wooden_cabinet',(2.65,.83,0),yaw=-90,scale=.65)
    top=.6801426*.65
    f=_Frame(b,(2.65,.83,0),10)
    f.box('television_stand_base',(0,0,top+.010),(.29,.31,.020),'black',.005)
    f.box('television_stand_neck',(.025,0,(top+.79)/2),(.055,.08,.79-top),'black',.004)
    f.box('television_frame',(0,0,1.105),(.048,1.20,.65),'black',.012)
    f.box('television_screen',(-.029,0,1.105),(.008,1.155,.605),'screen_black',.006,collision=False)
    b.cylinder('living_side_table_top',(-1.23,.83,.51),.19,.035,'walnut')
    for i in range(3):
        a=i*2*math.pi/3
        b.cylinder('living_side_table_leg'+str(i),(-1.23+.12*math.cos(a),.83+.12*math.sin(a),.246),.017,.492,'walnut')
    _ph(b,'living_plant','potted_plant_01',(2.40,1.76,0),scale=.70)
    b.box('living_wall_art_frame',(-.28,rear-.035,1.72),(1.35,.035,.65),'walnut',.006,collision=False)
    b.box('living_wall_art_canvas',(-.28,rear-.058,1.72),(1.30,.010,.60),'linen_ivory',.003,collision=False)
    for i in range(7):b.box('living_art_rib'+str(i),(-.79+i*.17,rear-.067,1.72),(.07,.007,.40-.035*abs(i-3)),'fabric',.015,collision=False)


def _mechanical_details(b):
    rear=ROOM_SPECS['mechanical_assembly']['bounds'][3]
    # A bolted bench vise has a genuine open jaw gap and screw, rather than a
    # solid surrogate across its opening. It sits on the perimeter service top.
    x,y,z=-.30,rear-.48,.916
    b.box('service_vise_base',(x,y,z+.014),(.23,.18,.028),'workshop_blue',.007)
    b.box('service_vise_slide',(x,y,z+.052),(.19,.065,.048),'steel',.003)
    for side in (-1,1):
        xx=x+side*.049
        b.box('service_vise_jaw'+str(side),(xx,y,z+.092),(.028,.13,.064),'workshop_blue',.004)
        b.box('service_vise_face'+str(side),(x+side*.032,y,z+.111),(.006,.12,.025),'steel',.001)
        for yy in (-.064,.064):
            b.cylinder('service_vise_anchor'+str(side)+'_'+str(yy),(x+side*.087,y+yy,z+.031),.008,.006,'chrome')
    b.cylinder('service_vise_screw',(x+.125,y,z+.052),.009,.16,'chrome',(0,90,0))
    b.cylinder('service_vise_handle',(x+.204,y,z+.052),.005,.15,'chrome',(90,0,0))
    # Service log and emergency-stop station are mounted on the wall, outside
    # the robot work volume. Their physical backing is kept in the collision set.
    b.box('workshop_log_back',(.79,rear-.077,1.93),(.32,.020,.31),'black',.005)
    b.label('workshop_service_log','SERVICE LOG\nINSPECT  /  CLEAN\nTORQUE  /  RECORD',(.79,rear-.088,1.93),(.29,.27),rotation=(90,0,0),font_size=40)
    b.box('workshop_stop_station',(1.30,rear-.044,1.16),(.13,.08,.18),'room_safety_yellow',.009)
    b.cylinder('workshop_stop_button',(1.30,rear-.102,1.18),.031,.045,'red',(90,0,0))
    b.box('workshop_task_light_body',(0,rear-.12,2.225),(1.12,.13,.040),'steel',.005)
    for i,x in enumerate((-.45,.45)):
        b.box('workshop_task_light_bracket'+str(i),(x,rear-.035,2.225),(.035,.070,.040),'steel',.002)
    b.box('workshop_task_light_diffuser',(0,rear-.12,2.202),(1.04,.085,.006),'room_light_glow',.001,collision=False)
    light(b,'service_counter_task',(0,rear-.14,2.195),(0,rear-.45,.92),(1.0,.07),13000.,(1,1,1),role='task',temperature=4600.)


def _laboratory_details(b):
    rear=ROOM_SPECS['laboratory']['bounds'][3]
    # The hood has an actual internal luminaire. Its source is below the header,
    # not hidden inside the solid housing or used as an unmotivated room fill.
    b.box('hood_task_diffuser',(-.76,rear-.34,2.004),(.95,.18,.012),'room_light_glow',.002,collision=False)
    light(b,'hood_task',(-.76,rear-.34,1.994),(-.76,rear-.35,.89),(.90,.15),15000.,(1,1,1),role='task',temperature=5000.)
    b.cylinder('hood_airflow_indicator',(-.24,rear-.641,2.09),.010,.006,'green',(90,0,0),False)
    b.label('hood_airflow_label','AIRFLOW',(-.40,rear-.643,2.09),(.20,.035),rotation=(90,0,0),font_size=28)
    # Gloves and a closed sample transport box occupy the clear counter beside
    # the hood, with their lowest surfaces on its measured procedural top.
    for n,x,y,w,d,h,mat in [('lab_glove_box',.25,rear-.40,.22,.14,.075,'lab_teal'),
                           ('lab_sample_box',.30,rear-.20,.24,.15,.11,'white')]:
        b.box(n,(x,y,.89+h/2),(w,d,h),mat,.005)
        b.label(n+'_label','NITRILE' if 'glove' in n else 'SAMPLES',(x,y-d/2-.001,.89+h/2),(w*.8,h*.55),rotation=(90,0,0),font_size=30)
    b.box('lab_glove_slot',(.25,rear-.40,.9653),(.13,.018,.0005),'black',.004,collision=False)
    # Pump bottle beside the basin, kept off the bowl opening.
    f=_Frame(b,(-1.30,.32,0),90)
    f.lathe('lab_soap_bottle',(-.40,-.03,.87),[(0,0),(.033,0),(.033,.105),(.020,.13),(.014,.13),(0,.13)],'white',collision=True)
    f.cylinder('lab_soap_pump',(-.40,-.03,1.015),.011,.030,'lab_teal')
    f.tube('lab_soap_spout',[(-.40,-.03,1.03),(-.40,-.075,1.03),(-.40,-.075,1.02)],.005,'lab_teal',collision=True)
    b.box('lab_towel_dispenser',(-1.565,.88,1.20),(.070,.24,.27),'white',.009)
    b.box('lab_towel_slot',(-1.528,.88,1.08),(.002,.18,.018),'black',.002,collision=False)


def _kitchen_details(b):
    rear=ROOM_SPECS['kitchen']['bounds'][3]
    side=_Frame(b,(-1.315,.66,0),90)
    # Drainboard and separated plate slots leave the sink and worktable clear.
    side.box('dish_rack_drainboard',(.15,0,.891),(.32,.32,.012),'steel',.006)
    for i in range(7):
        xx=.02+i*.040
        side.tube('dish_rack_wire'+str(i),[(xx,-.13,.904),(xx,-.13,1.02),(xx,.13,1.02),(xx,.13,.904)],.003,'chrome',collision=True)
    for i in range(3):
        side.cylinder('dish_rack_plate'+str(i),(.04+i*.08,0,.971),.074,.008,'ceramic',(0,90,0))
    for i in range(2):
        xx=-1.43+i*.105
        b.lathe('kitchen_cleaning_bottle'+str(i),(xx,rear-.26,.885),[(0,0),(.032,0),(.032,.12),(.018,.145),(.013,.145),(0,.145)],'sage_enamel' if i==0 else 'white',collision=True)
        b.cylinder('kitchen_cleaning_pump'+str(i),(xx,rear-.26,1.044),.011,.028,'chrome')
    # Two local task sources match the actual upper cabinet and extractor.
    b.box('kitchen_cabinet_task_diffuser',(1.28,rear-.20,1.6005),(.49,.23,.009),'room_light_glow',.002,collision=False)
    light(b,'kitchen_counter_task',(1.28,rear-.20,1.592),(1.28,rear-.40,.885),(.46,.20),10000.,(1,1,1),role='task',temperature=4000.)
    b.box('extractor_task_diffuser',(.64,rear-.32,1.585),(.37,.17,.010),'room_light_glow',.002,collision=False)
    light(b,'range_task',(.64,rear-.32,1.576),(.64,rear-.35,.90),(.34,.14),8000.,(1,1,1),role='task',temperature=3800.)
    # A hollow ceramic utensil crock; handles are supported by its real bottom.
    b.lathe('kitchen_utensil_crock',(1.30,rear-.31,.885),[(0,0),(.046,0),(.047,.12),(.041,.12),(.040,.008),(0,.008)],'ceramic',collision=True)
    for i in range(3):
        x=1.275+i*.023
        b.cylinder('kitchen_utensil_handle'+str(i),(x,rear-.31,1.003),.004,.22,'oak')
        b.box('kitchen_utensil_head'+str(i),(x,rear-.31,1.129),(.018,.008,.032),'oak',.005)


def _study_details(b):
    rear=ROOM_SPECS['study']['bounds'][3];y=rear-.225
    for bay,x in enumerate([-.72,.48,1.68]):
        # Weighted bookends stand beside the books, not interpenetrating a foot
        # under books whose original positions must remain unchanged.
        shelf=2 if bay%2==0 else 3;z=[.11,.68,1.07,1.46,1.85][shelf]
        end=x-.545+sum(.031+.006*((i+bay+shelf)%4) for i in range(13))
        b.box('library_bookend'+str(bay),(end+.009,y-.015,z+.080),(.018,.15,.13),'steel',.003)
        b.label('library_shelf_label'+str(bay),['REFERENCE','LITERATURE','DESIGN'][bay],(x,y-.196,.68),(.20,.020),rotation=(90,0,0),font_size=26)
    # A wall clock supplies a stable visual landmark in side and wrist views.
    x=ROOM_SPECS['study']['bounds'][1]-.0175;yy=.95;z=1.91
    b.cylinder('study_clock_frame',(x,yy,z),.145,.035,'walnut',(0,90,0))
    b.cylinder('study_clock_face',(x-.019,yy,z),.132,.003,'paper',(0,90,0),False)
    for i in range(12):
        a=math.pi*i/6
        b.tube('study_clock_tick'+str(i),[(x-.022,yy+.113*math.sin(a),z+.113*math.cos(a)),
                                        (x-.022,yy+.125*math.sin(a),z+.125*math.cos(a))],.0016,'black')
    for n,dy,dz in [('hour',-.060,.035),('minute',.086,.050)]:
        b.tube('study_clock_'+n,[(x-.025,yy,z),(x-.025,yy+dy,z+dz)],.0024,'black')


def _card_details(b):
    rear=ROOM_SPECS['card_room']['bounds'][3]
    commode=next(record for record in b.records if record.get('name')=='tea_chinese_commode')
    z=commode['position'][2]+commode['bounds_size'][2]
    # Use the referenced furniture's actual grounded height, not an assumed
    # cabinet height. Tea storage occupies the rear sideboard, not the Go board.
    for i,x in enumerate([-.30,-.14,.02]):
        b.lathe('tea_storage_canister'+str(i),(x,rear-.30,z),[(0,0),(.052,0),(.052,.14),(.045,.15),(.045,.153),(0,.153)],'tea_porcelain',collision=True)
        b.cylinder('tea_storage_lid'+str(i),(x,rear-.30,z+.158),.050,.010,'walnut')
        b.label('tea_storage_label'+str(i),['绿茶','乌龙','红茶'][i],(x,rear-.353,z+.075),(.065,.065),rotation=(90,0,0),font_size=45)
    b.box('tea_service_tray',(.82,rear-.30,z+.009),(.36,.25,.018),'walnut',.008)
    for i in range(3):
        x=.71+i*.11
        b.lathe('tea_service_cup'+str(i),(x,rear-.30,z+.018),[(0,0),(.027,0),(.037,.043),(.033,.043),(.023,.005),(0,.005)],'tea_porcelain',collision=True)
    b.box('tea_scroll_light_body',(-1.4125,.62,2.31),(.075,.44,.034),'walnut',.004)
    light(b,'tea_scroll_accent',(-1.34,.62,2.286),(-1.41,.62,1.78),(.38,.045),5500.,(1,1,1),role='accent',temperature=2700.)


def _living_details(b):
    x,y=2.18,.23
    # Reading lamp has a weighted foot, continuous stem, open shade and power
    # cord; all solid components keep collision geometry outside the workcell.
    b.cylinder('living_floor_lamp_base',(x,y,.018),.14,.036,'black')
    b.cylinder('living_floor_lamp_stem',(x,y,.738),.012,1.404,'chrome')
    b.lathe('living_floor_lamp_shade',(x,y,1.39),[(.17,0),(.172,.008),(.12,.20),(.115,.20),(.164,.008),(.163,0)],'linen_ivory',collision=True)
    for side in (-1,1):
        b.tube('living_floor_lamp_spoke'+str(side),[(x,y,1.435),(x+side*.166,y,1.405)],.003,'chrome',collision=True)
    b.sphere('living_floor_lamp_bulb',(x,y,1.43),.025,'room_light_glow')
    b.tube('living_floor_lamp_cord',[(x+.05,y,.003),(x+.21,y,.003),(2.81,y,.003),(2.825,y,.09)],.003,'black',collision=True)
    b.box('living_floor_lamp_plug',(2.835,y,.09),(.030,.04,.035),'black',.003)
    light(b,'living_reading_task',(x,y,1.382),(1.52,1.30,.70),(.27,.27),18000.,(1,1,1),shape='disk',role='task',temperature=3000.)
    top=.6801426*.65
    for i,yy in enumerate([.24,1.42]):
        b.box('living_media_speaker'+str(i),(2.65,yy,top+.10),(.13,.12,.20),'black',.006)
        b.cylinder('living_speaker_cone'+str(i),(2.583,yy,top+.095),.039,.005,'fabric',(0,90,0),False)
        b.cylinder('living_speaker_tweeter'+str(i),(2.583,yy,top+.16),.012,.005,'steel',(0,90,0),False)
    b.box('living_side_magazine',(-1.23,.83,.535),(.19,.14,.015),'paper',.002)
    b.label('living_side_magazine_cover','AT HOME',(-1.23,.83,.5428),(.15,.055),font_size=38)


def build_functional_details(b,family):
    """Add perimeter uses without moving existing geometry or occupying the workcell."""
    {'mechanical_assembly':_mechanical_details,'laboratory':_laboratory_details,
     'kitchen':_kitchen_details,'study':_study_details,'card_room':_card_details,
     'living_room':_living_details}[family](b)


def _inventory(b,family,paths):
    """Explicit semantic assemblies for independent actual-mesh layout auditing."""
    groups={
        'mechanical_assembly':[
            ('service_counter',('service_base','service_oak_top','service_back_lip')),
            ('storage_shelf',('assembly_shelf',))],
        'laboratory':[
            ('lab_counter_and_hood',('clean_casework','lab_epoxy_counter','hood_')),
            ('clean_storage',('clean_storage_',)),('wash_cabinet',('lab_wash_',))],
        'kitchen':[
            ('back_casework',('kitchen_base','kitchen_counter','kitchen_sink','kitchen_tap')),
            ('return_casework',('kitchen_return',)),('cooking_range',('kitchen_real_stove','stove_leveling')),
            ('refrigerator',('fridge_',))],
        'study':[('built_in_library',('library_bay','library_crown')),('window_reading_bench',('reading_bench_',))],
        'card_room':[('tea_storage',('tea_chinese_commode',)),('tea_chair_left',('tea_left_chair',)),('tea_chair_right',('tea_right_chair',))],
        'living_room':[('sofa',('living_sofa',)),('armchair',('living_armchair',)),('media_console',('living_console','television_')),
                       ('reading_lamp',('living_floor_lamp_',)),
                       ('side_table',('living_side_table',)),('lounge_plant',('living_plant',))],
    }[family]
    assigned=set()
    for name,prefixes in groups:
        subset=[p for p in paths if p.rsplit('/',1)[-1].startswith(prefixes)]
        assigned.update(subset)
        for p in subset:b.stage.GetPrimAtPath(p).SetCustomDataByKey('scence:layoutRole','floor_furniture')
        b.records.append({'kind':'interior_group','name':name,'role':'floor_furniture','prim_paths':subset})
    # Remaining decor stays explicit, rather than contaminating floor-clearance tests.
    wall=('workshop_','peg_','service_hook','service_spanner','extractor_','kitchen_upper',
          'kitchen_spice','spice_bracket','tea_scroll','tea_calligraphy','living_wall_art','living_art',
          'study_clock','lab_towel','kitchen_cabinet_task')
    for role,selection in [('wall_fixture',[p for p in paths if p not in assigned and p.rsplit('/',1)[-1].startswith(wall)]),
                           ('architectural',[p for p in paths if p.rsplit('/',1)[-1].startswith('living_seating_rug')])]:
        assigned.update(selection)
        for p in selection:b.stage.GetPrimAtPath(p).SetCustomDataByKey('scence:layoutRole',role)
        b.records.append({'kind':'interior_group','name':family+'_'+role,'role':role,'prim_paths':selection})
    remainder=[p for p in paths if p not in assigned]
    for p in remainder:b.stage.GetPrimAtPath(p).SetCustomDataByKey('scence:layoutRole','supported_decor')
    b.records.append({'kind':'interior_group','name':family+'_supported_decor','role':'supported_decor','prim_paths':remainder})


def build_interior(b,family):
    _materials(b)
    root=b.stage.GetPrimAtPath('/World/Geometry')
    before={str(p.GetPath()) for p in root.GetChildren()}
    {'mechanical_assembly':_mechanical,'laboratory':_laboratory,'kitchen':_kitchen,
     'study':_study,'card_room':_card,'living_room':_living}[family](b)
    build_functional_details(b,family)
    _inventory(b,family,[str(p.GetPath()) for p in root.GetChildren() if str(p.GetPath()) not in before])
