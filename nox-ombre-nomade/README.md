# NOX Parfums — Ombre Nomade · Recreación audiovisual

Este Reel vertical (1080×1920, 30 fps, 43,9 s) recrea el lenguaje visual de la publicidad de referencia: un Reel ASMR que muestra "cómo se prepara el perfume". El producto original se reemplaza por el frasco **NOX Ombre Nomade**, reconstruido a partir de las fotografías del producto.

Todo es procedural y se genera desde código: fondo, vidrio, líquido, física, utilería, manos, tipografía y sonido. No se usa material de la referencia en el video final.

- Video final: `exports/NOX_OMBRE_NOMADE.mp4` (también copiado a `/exports/NOX_OMBRE_NOMADE.mp4`)
- Escenas individuales: `exports/scenes/NN_<plano>.mp4`
- Banda sonora: `exports/NOX_OMBRE_NOMADE_audio.wav`

## Cómo renderizar

```bash
pip install numpy opencv-python-headless scipy pillow
python3 src/render.py                 # simulación + render de las 27 escenas (unos 9 min con 4 núcleos)
python3 src/finalize.py               # sonido + concatenación + máster para Instagram
python3 src/render.py --frames 90,460,1300   # fotogramas sueltos para revisar (build/preview)
```

## Estructura

| Archivo | Función |
|---|---|
| `src/timeline.py` | Lista de planos con los tiempos de corte medidos, niveles y color del líquido. También contiene el *director*, que dirige cada plano |
| `src/sim.py` | Fluidos 2D *stable fluids* (humo, tinturas tipo tinta en agua, turbidez y schlieren), partículas y burbujas |
| `src/bottle.py` | El frasco: refracción, paredes y base gruesas con reflexión interna, menisco, cáusticas, atomizador cromado, tubo sonda y bandeja negra |
| `src/props.py` | Pipetas, goteros, embudos, jeringa, frasco ámbar, pinzas, vaso, crimpadora, guantes de nitrilo, chorros, humo y bruma del spray. También genera los ingredientes (oud, astillas, botánicos, cardamomo y azafrán) |
| `src/background.py` | Bokeh de jardín al atardecer, mesa de madera y luz moteada |
| `src/post.py` | Bloom, curva fílmica, etalonaje cálido, viñeta, grano y tipografía NOX |
| `src/audio.py` | Diseño sonoro original sincronizado con la lista de planos |
| `src/finalize.py` | Concatenación, mezcla de audio y máster H.264 (High 4.2, BT.709, AAC 320k, −14 LUFS, faststart) |
| `assets/product/` | Fotografías del producto usadas como referencia geométrica |
| `assets/fonts/` | Cinzel, Cormorant Garamond y Montserrat (licencia OFL) |
| `analysis/` | Hojas de contacto de la referencia, cortes medidos (`scene_cuts.json`) y este storyboard |

## Análisis de la referencia

La referencia es una grabación de pantalla de un Reel de Instagram (384×832, 60 fps, 43,925 s). Los cortes se midieron por diferencia entre fotogramas a 60 fps (ver `analysis/scene_cuts.json` y `analysis/frames/`). La estructura es la siguiente:

- **0–36,6 s:** cámara fija en trípode, frontal, a la altura de los hombros del frasco. Fondo de follaje desenfocado iluminado por el sol y madera cálida detrás de la base. Cada ingrediente entra con un *jump cut* y el nivel del líquido salta entre planos.
- **36,6–41,6 s:** cámara en mano. El frasco se inclina y se agita, luego aparece en una mesa con luz moteada y sigue un barrido rápido con desenfoque de movimiento.
- **41,6–43,9 s:** cierre del producto (etiqueta y tapa en el original) y plano *hero* final.

## Storyboard técnico

| # | Tiempo (s) | Referencia | Recreación NOX | Técnica |
|---|---|---|---|---|
| 01 | 0,00–0,40 | Humo de incienso dentro del frasco vacío, oud en el fondo | Igual; nota **AHUMADO** | Fluido en modo aire con flotabilidad negativa y pre-roll de 2,5 s; volutas exteriores con ruido rizado |
| 02 | 0,40–0,92 | El humo se disipa | Igual | Decaimiento del canal de humo |
| 03 | 0,92–2,15 | Pipeta de vidrio, chorro fino | Igual | Chorro con refracción y bordes especulares; burbujas al impactar |
| 04 | 2,15–4,15 | Embudo con aceite ámbar | Igual; **ÁMBAR** | Embudo de vidrio con contenido que baja; tintura ámbar inyectada con chorro |
| 05 | 4,15–6,15 | Jeringa con inyección violenta y espuma | Igual; **BALSÁMICO** | Chorro de alta velocidad, turbidez, schlieren y unas 400 burbujas |
| 06 | 6,15–6,68 | Pipeta | Igual | — |
| 07 | 6,68–9,08 | Embudo con astillas de madera | Igual; **OUD** | Partículas de astillas con gravedad, arrastre y acoplamiento al fluido |
| 08 | 9,08–11,08 | Mano con guante vierte botánicos desde un frasco | Igual; **LEÑOSO** | Frasco ámbar y guante; las escamas flotan y luego se hunden |
| 09 | 11,08–13,25 | Gotero transparente | Igual | Schlieren en la punta |
| 10 | 13,25–16,12 | Gotero turquesa | Igual | Pluma de tinta con flotabilidad positiva y confinamiento de vorticidad |
| 11 | 16,12–18,88 | Gotero naranja/rojo | Igual; **AFRUTADO** | Ídem |
| 12 | 18,88–21,63 | Gotero rojo profundo | Igual; **ROSA** | Se forma una capa de sedimento en el fondo |
| 13 | 21,63–24,22 | Gotero marrón | Igual; **CUERO** | Ídem |
| 14 | 24,22–26,95 | Pipeta larga amarilla | Igual; **DULCE** | Columna amarilla que se vacía |
| 15 | 26,95–29,25 | Pinzas sueltan cardamomo (4 *jump cuts*) | Igual; **PICANTE CÁLIDO** | Sub-cortes en 27,55, 28,28 y 28,72 |
| 16 | 29,25–29,78 | Gotero | Igual | — |
| 17 | 29,78–32,78 | Embudo con azafrán | Igual | Hebras de azafrán como partículas que tiñen el líquido |
| 18 | 32,78–35,05 | Vaso de precipitado vierte alcohol hasta llenar | Igual. El líquido se diluye hasta el **dorado pálido real** del producto | Chorro en arco, turbulencia, mezcla acelerada de tinturas |
| 19 | 35,05–35,55 | Pipeta de enrase | Igual | — |
| 20 | 35,55–36,58 | Mano coloca el atomizador; herramienta blanca de crimpado | Atomizador **plateado real** con tubo sonda; crimpado | Metal cromado con mapa de entorno |
| 21 | 36,58–37,12 | Plano frontal en reposo | Igual | Leve acercamiento |
| 22 | 37,12–38,58 | Mano inclina y agita el frasco (cámara en mano) | Igual | Frasco rotado sobre un fondo nivelado; superficie con resorte de oleaje y conservación de volumen; partículas en suspensión |
| 23 | 38,58–40,65 | Frasco en mesa con luz moteada, mano sobre el pulsador | Igual | Patrón de sombras de hojas sobre mesa, vidrio y metal |
| 24 | 40,65–41,55 | Barrido rápido en mano | Igual | 9 subfotogramas por frame + desenfoque direccional |
| 25 | 41,55–42,42 | Aplicación de la etiqueta (plano inclinado) | **El frasco baja a su bandeja negra.** El frasco real no tiene etiqueta y no se inventa una | Bandeja negra mate medida en la foto |
| 26 | 42,42–43,02 | *Hero* frontal, la mano sale | Igual, con la bandeja | — |
| 27 | 43,02–43,93 | Se coloca la tapa dorada | **El dedo pulsa el atomizador y la bruma sale al sol.** Aparece el título **NOX · PARFUMS · Ombre Nomade** | Unas 2600 gotas de spray; tipografía animada |

Las notas que aparecen como subtítulos son las diez impresas en la caja NOX: ámbar, oud, picante cálido, rosa, ahumado, leñoso, afrutado, cuero, balsámico y dulce.

## Fidelidad del producto

Las proporciones se midieron en `ombre_nomade_front.jpg`, con 630 px de vidrio equivalentes a 960 px de diseño:

- Cuerpo rectangular de proporción 1,7:1, con esquinas de radio pequeño, paredes laterales gruesas, base pesada y un resalte circular de vidrio alrededor del cuello.
- Atomizador plateado: virola con faldón (222 px), vástago (158 px) y pulsador cilíndrico (184×168 px) con el punto blanco de la boquilla al frente.
- Tubo sonda transparente que se curva hacia la esquina inferior izquierda, como en la foto.
- Bandeja negra mate de 990×297 px, que oculta aproximadamente la cuarta parte inferior del frasco.
- Líquido final dorado pálido, como en las fotos. El ámbar oscuro de la referencia se usa solo durante la preparación.

El frasco no lleva etiqueta porque el producto real no la tiene. La identidad NOX se aplica con tipografía en pantalla y no sobre el vidrio. Tampoco se muestra la ilustración de la caja.
