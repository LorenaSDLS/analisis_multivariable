# Distancia-360: Cómputo Masivo de Similitud entre Municipios

Se implementa el pipeline completo para el cálculo de distancias par-a-par entre todos los municipios de México (N=2475), por medio del uso de series de tiempo climáticas (temperatura, precipitación, evaporación y radiación) a resolución diaria por 10 años, además de variables como unidades climáticas, edafología, topoforma. 

Obteniendo una matriz de similitud de 2475 x 2475 municipios, los resultados se almacenan en formato HDF5 y se usan para construir un grafo k-NN nacional.  

Se incluyen perfiles de rendimiento, validaciones de integridad y scripts reproducibles.


## Objetivos del Proyecto

1. **Calcular la matriz general de similitud** entre todos los pares de municipios. 
Esto se realiza primero obteniendo las matrices de similitud de cada variable por medio de: 

   - *Distancia euclidiana multivariante* sobre las series temporales
   - *Similitud proporcional* para las variables numéricas
   - *Distancia de *Levenshtein* para las variables categóricas

Y luego se saca el promedio de cada índice para obtener la matriz general de similitud.

2. **Construir un grafo k-Nearest Neighbors (k-NN)**  
   Donde cada municipio se conecta con sus k vecinos más similares (peso = similitud).

3. **Generar scripts reproducibles y escalables**, incluyendo:
   - Cómputo por bloques (block-wise)
   - Opcional: paralelización
   - Medición de tiempo y memoria
   - Validaciones de integridad
   - Exportación en formatos adecuados (HDF5, Parquet)

## Dependencias

Recomendado crear un entorno virtual con python 3.12

```bash
pyenv local 3.12.7
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
