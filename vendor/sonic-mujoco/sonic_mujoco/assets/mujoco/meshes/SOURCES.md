# Scene mesh sources

The following visual meshes and base-color textures are derived from free
Sketchfab downloads licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

- `avocado_plush_scan.obj` and `../textures/avocado_plush_scan.png`:
  [Avocado Plush Toy](https://sketchfab.com/3d-models/avocado-plush-toy-567c19c1347542c3a63eabc8675257ce)
  by [Tim (melsto)](https://sketchfab.com/melsto).
- `water_gallon_20l.obj`:
  [water gallon](https://sketchfab.com/3d-models/water-gallon-2b24900b634645649df674983b9434b1)
  by [AnshiNoWara](https://sketchfab.com/anshinowara).
- `folding_table.obj` and `../textures/folding_table_basecolor.png`:
  [Folding Tables](https://sketchfab.com/3d-models/folding-tables-9a483f7f5f514d9a85af06a63c056bb8)
  by [Jesus Fernandez Garcia](https://sketchfab.com/jamyzgenius).
- `upholstered_chair.obj` and
  `../textures/upholstered_chair_basecolor.png`:
  [Modern Upholstered Armchair](https://sketchfab.com/3d-models/modern-upholstered-armchair-4096px2-fd402ff0f1654eb4b5ec20a832b6895c)
  by [Mark Peters](https://sketchfab.com/mark-peters).
- `interior_door.obj` and `../textures/interior_door_basecolor.png`:
  [Interior Wooden Door](https://sketchfab.com/3d-models/interior-wooden-door-720c98140848429c985fbd7c77eac70e)
  by [joshtmc](https://sketchfab.com/joshtmc).

Meshes were converted to OBJ, aligned to the MuJoCo body frames, and scaled to
the scene reference dimensions. Large base-color maps were reduced to 1024 px.
The downloaded meshes are visual-only; compact MuJoCo primitives and flexes
remain responsible for collision, mass, compliance, and tactile data.
