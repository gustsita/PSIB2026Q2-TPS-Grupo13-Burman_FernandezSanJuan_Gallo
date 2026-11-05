# Decisiones del TPS – Grupo 13

Este documento explica cada decisión del trabajo: qué se decidió, por qué y con qué evidencia. Sirve para escribir el informe y preparar la defensa. Cubre de la Parte 0 a lo hecho hasta ahora en la Parte 3. El detalle y las figuras están en los notebooks.

## La idea del trabajo

Queremos saber si un EEG portátil de 4 canales puede detectar cambios en el nivel de activación cerebral, en especial en la banda beta (13–30 Hz). Comparamos:

- 1-back contra 3-back (carga cognitiva);
- sin música contra música relajante, estimulante y generada por IA (experimento 1);
- basal contra perfume y café (experimento 2).

También queremos ver si lo que el artículo muestra en un solo sujeto por experimento se repite en todos, y si la beta se relaciona con el desempeño.

## La cadena de procesamiento

```
EEG crudo de una sesión
  1. Marcar muestras saturadas (±1000 µV) y ensanchar la marca 1 s
  2. LMS: cancelar la interferencia de 60 Hz
  3. Pasa banda Butterworth con filtfilt
       - 1-40 Hz  para delta, theta y alfa
       - 13-40 Hz para decidir qué épocas sirven para beta
  4. Recortar en épocas de 2 s (un ensayo cada una)
  5. Rechazar épocas con artefactos y excluir canales con picos angostos
  6. PSD de Welch por época y potencia por banda
  7. Promediar por condición y comparar dentro de cada sujeto
```

## Parte 0 – Datos y sincronización

**Frecuencia de muestreo: se usan 256 Hz fijos, no los timestamps.** Los timestamps están redondeados al milisegundo, así que el intervalo entre muestras parece valer 3 o 4 ms, cuando el real es 3,906 ms. Calculada como cantidad de muestras sobre duración, la frecuencia da entre 256,0 y 256,2 Hz en los 20 sujetos, sin huecos.

**Sujeto A8: se estimó un tag y no se descartó al sujeto.** Le falta el inicio de la sesión 2 (tiene 7 tags en lugar de 8).
- Los tags se asignan a cada sesión por su hora, no de a pares.
- El inicio faltante se estima como el primer estímulo menos 11,8 s, que es la mediana de ese margen en las 66 sesiones con sus dos tags.
- Se usa la mediana y no la media (17,4 s) porque hay márgenes de hasta 90 s que la desplazan.
- No se descarta porque el tag solo delimita la sesión: los bloques se ubican con la hora de cada estímulo, que está completa.

**Aciertos: se unifican dos columnas.** El sujeto puede responder durante el estímulo o durante la cruz de fijación, y E-Prime lo guarda en columnas distintas. Mirando una sola, A9 parecía acertar el 31 %; en realidad acierta entre 78 y 95 %, según el tipo de bloque.

**Pausas: duran unos 15 s, no los 10 s del anteproyecto.** Es el valor medido en los datos. Por eso el ciclo tarea-pausa dura unos 58 s, lo que importa para la Parte 4.

## Parte 1 – Filtrado

**Por qué hay filtro adaptativo si después hay un pasa banda.** El pico de 60 Hz es fuerte en algunos canales: en B1, TP9 tiene una PSD en 60 Hz de 444 µV²/Hz cruda, 0,073 solo con pasa banda y 0,0013 con LMS más pasa banda. Además, filtros óptimos es un tema obligatorio de la consigna.

**Por qué LMS y referencia sintética.** No hay un canal que mida solo la interferencia, pero sabemos su frecuencia. Con un coseno y un seno de 60 Hz y dos pesos se puede representar una senoidal de cualquier amplitud y fase. El filtro se adapta a medida que la interferencia cambia, y por eso conviene frente a un filtro fijo.

**Por qué LMS antes del pasa banda.** Si se filtrara primero, el LMS casi no tendría interferencia que cancelar ni se podría evaluar.

**Por qué μ = 3·10⁻³.**
- Se hizo un barrido sobre el EEG real.
- Con μ ≤ 10⁻³ queda interferencia en los canales donde es fuerte.
- Con μ ≥ 5·10⁻³ el filtro se ensancha y se pierde más del 10 % de la potencia a ±0,5 Hz de 60 Hz.
- Con el valor elegido, la constante de tiempo es τ = 1/μ = 333 muestras, unos 1,3 s, y la pérdida a ±0,5 Hz es de 4 a 7 %.
- Primero se verificó con una señal sintética: los pesos convergieron a los valores esperados (61,2 y −51,5).

**Por qué Butterworth de 1–40 Hz, orden 4, con filtfilt.**
- *Butterworth:* su banda de paso es plana, así que no distorsiona la amplitud relativa entre bandas.
- *1 Hz:* saca la deriva lenta. Por eso delta se toma desde 1 Hz, y no desde 0,5 Hz. El artículo también filtra desde 1 Hz.
- *40 Hz:* deja margen sobre beta y atenúa lo que quede de 60 Hz.
- *Orden 4:* es el menor que deja plana la banda de 2 a 30 Hz (pérdida menor a 0,5 dB) y atenúa 38 dB en 60 Hz. El orden 2 pierde casi 2 dB en 30 Hz.
- *filtfilt:* filtra ida y vuelta, así que no hay retardo que corra la señal respecto de los bloques.

**Por qué un solo filtro.** La profe nos pidió no comparar filtros y profundizar en otras partes. Por eso se sacaron la comparación FIR contra IIR, Wiener y el notch.

**Residuo en A3-TP10.** En unos pocos canales el pico de 60 Hz no baja del todo, porque la interferencia está modulada cada 5 s y el LMS no llega a seguirla. Lo que queda es como máximo el 0,06 % de la potencia beta, así que no afecta el análisis.

## Parte 2 – Épocas y artefactos

**Por qué épocas de 2 s.** Los estímulos se presentan cada 2 s, así que cada época es un ensayo completo. Además, una época más larga se pierde entera por un solo artefacto.

**Por qué la saturación se marca antes de filtrar.** La Muse satura en ±1000 µV. El filtrado reparte ese salto entre las muestras vecinas y después ya no se detecta. Por eso se marca sobre la señal cruda y la marca se ensancha 1 s para cada lado.

**Umbrales de rechazo.** Salen de las distribuciones de los 20 sujetos:

| Criterio | Umbral | Por qué |
|---|---|---|
| Saturación | ±1000 µV, ±1 s | Límite del equipo |
| Pico a pico | > 150 µV | El valle entre los dos modos del histograma cae en unos 140 µV |
| Varianza | z robusto > 3,5 | Valor habitual (Iglewicz y Hoaglin); se calcula por sujeto y canal porque cada electrodo seco tiene su contacto |
| Curtosis | > 5 | Una señal gaussiana da 3; solo superan 5 las épocas con eventos aislados |
| Señal plana | desvío < 1 µV | Muy por debajo del percentil 5; electrodo sin contacto |

**Por qué hay dos niveles de rechazo.** TP9 y TP10 tienen deflexiones de cientos de µV en 16 de los 20 sujetos, compatibles con parpadeos vistos a través de la referencia Fpz. Rechazarlas para todas las bandas dejaría esos canales casi sin datos. Entonces:

- `usable`: criterios sobre 1–40 Hz. Se usa para delta, theta y alfa.
- `usable_beta`: criterios sobre 13–40 Hz. Se usa para beta.

Con esto se conserva el 86 % de las épocas para beta y el 45 % para las otras bandas. Por eso **delta, theta y alfa se analizan en los canales frontales**: TP9 y TP10 sirven para esas bandas solo en 3 sujetos.

**Por qué se excluyen canales por picos angostos.** Hay picos en 21,3 y 24 Hz, dentro de beta, de hasta 1000 veces el fondo. Son casi senoidales, así que los criterios por época no los detectan. Se mide qué fracción de beta aportan: la distribución tiene dos grupos (hasta 16 % y desde 30 %), y el umbral de 25 % los separa sin casos dudosos. Se excluyen A4-AF7, A9-AF8, B1-AF7, B1-TP9 y B5-AF7.

El origen no se pudo verificar. Una posibilidad es que la Muse transmite en paquetes de 12 muestras, y 256 / 12 = 21,33 Hz.

**Regla de canales.** Un canal se usa en un sujeto si conserva al menos el 50 % de sus épocas de tarea en cada sesión. Si no, las condiciones no se podrían comparar dentro del sujeto.

**El rechazo no sesga la comparación de carga.** El porcentaje de épocas usables no difiere entre 1-back y 3-back (Wilcoxon, p > 0,3).

## Parte 3 – Espectro

**Por qué la máscara de banda incluye los dos bordes.** Con `f < f_max`, el cálculo termina una frecuencia antes del límite. Con ruido blanco y resolución de 1 Hz, delta perdía 37 %, theta 26 %, alfa 21 % y beta 5 %. Con `f <= f_max`, las bandas se tocan sin huecos y su suma da la potencia total.

**Por qué segmentos de Welch de 2 s.**

| | 1 s | 2 s |
|---|---|---|
| Resolución | 1 Hz | 0,5 Hz |
| Senoidal de 12 Hz que se cuenta como beta | 8,33 % | 0 % |
| Error en delta (ruido blanco) | −4,7 % | −1,5 % |
| Variación de beta en una época | 19,6 % | 23,6 % |
| Variación de beta promediando 168 épocas | 1,39 % | 1,76 % |

No puede ser más largo que 2 s, porque se metería en el ensayo siguiente. El costo es más varianza por época, pero se compensa al promediar. El promedio de Welch sigue estando: en lugar de promediar segmentos dentro de una época, se promedian las épocas válidas de cada condición. En el EEG de A1, theta, alfa y beta cambian como mucho un 4 % entre las dos opciones, así que los resultados no dependen de esta decisión.

**Por qué la medida principal de beta es la potencia absoluta.** La potencia relativa divide por el total de 1 a 30 Hz, y ese total incluye las deflexiones. En TP9 de A1, la mediana de beta relativa pasa de 0,411 a 0,028 en las épocas con deflexión. Por eso:

- beta se mide con `beta_abs`;
- la potencia relativa y el cociente theta/beta se calculan solo en épocas `usable`, en la práctica en los canales frontales.

**Las deflexiones suben beta, pero se mantiene `usable_beta`.** En los 20 sujetos, las épocas con deflexión tienen más beta en TP9 (×1,15, p = 0,003) y TP10 (×1,21, p = 0,0005), y no significativamente en los frontales (×1,08). Se mantiene `usable_beta` por tres razones:

1. La proporción de épocas con deflexión no cambia entre 1-back y 3-back (80,1 % y 80,7 %, p = 0,10) ni entre sesiones (p = 0,77 y p = 0,50).
2. Con los mismos sujetos, sacar esas épocas cambia los cocientes entre condiciones un 6,4 % como máximo. Solo 5 o 6 sujetos permiten esta prueba.
3. Rechazarlas dejaría TP9 y TP10 casi sin datos, porque cerca del 80 % de las épocas tiene deflexión.

**Límites que quedan por esto:**
- No se compara el nivel de beta entre sujetos, solo dentro de cada uno.
- En TP9 y TP10, un efecto menor al 6 % no se distingue de este sesgo. Los frontales sirven de control.
- La profundidad de las deflexiones cambia en el experimento 2 (163 µV con perfume, 179 en basal, 191 con café; p = 0,002). Hay que tenerlo en cuenta al analizar el café en TP9 y TP10.

## Qué se corrigió del trabajo anterior

En la Parte 2 se cambió **solo texto**, no el código del rechazo:

| Dónde | Decía | Ahora dice |
|---|---|---|
| Notebook 02, título 2.1 | Las deflexiones "no afectan a beta" | "Afectan poco a beta" |
| Notebook 02, después del cálculo en B1 | "No cambian la de beta" | En B1 casi no la cambian; en los 20 sujetos, ×1,15 y ×1,21 |
| Notebook 02, conclusión 3 | "No cambian la de beta" | +15 % y +21 %, y casi no altera las comparaciones |
| `epocas.py`, explicación inicial | Igual que arriba | Igual que arriba |

**No cambiaron** los umbrales, `usable_beta`, los canales excluidos ni los resultados del notebook 02.

La medición original no estaba mal, pero se hizo en un solo sujeto, B1, que es de los que menos efecto tienen. Además, TP9 de B1 es un canal que después se excluye.

## Preguntas probables en la defensa

**¿Por qué no usaron ICA para sacar los parpadeos?** Con 4 canales hay muy pocas componentes para separar fuentes, y la profe nos pidió acotar el análisis. Es una decisión de criterio: no lo probamos con los datos.

**¿Por qué no detectan QRS ni analizan HRV?** El trabajo se centra en el EEG. Esos temas son propios del ECG, y este dataset no lo tiene. Trae pulso (PPG) de las pulseras, que permitiría un análisis parecido, pero quedó fuera del alcance.

**¿Son parpadeos de verdad?** No lo podemos asegurar, porque no hay canal de EOG. Lo que sabemos es que son simultáneos en TP9 y TP10, no están ligados al estímulo y son compatibles con parpadeos vistos a través de Fpz.

**¿Por qué A9 da distinto que en el artículo?** A9 tiene canales muy malos: en beta solo se usan TP9 y AF7, y el artículo mira TP9 y TP10. La reproducción de ese sujeto es parcial.

**¿Cómo saben que no es todo ruido?** El LMS y las funciones de potencia se probaron con señales de resultado conocido, y los umbrales se justifican con distribuciones medidas o valores habituales. Las comparaciones son dentro de cada sujeto y los canales frontales sirven de control.

## Pendientes

- Terminar la Parte 3: tabla de potencias por sujeto, condición y canal, y verificación con AR.
- Parte 4: periodograma de Lomb.
- Parte 5: estadística y relación entre beta y desempeño.
- Corregir la nota sobre AF8 en `epocas.py`: decía que podía ser mal contacto, pero en B1 es de los canales más limpios.
- Mergear la rama de la Parte 2 en `main`.
