# SAMI
Módulo de Detección Facial y Persistencia de Datos para SAMI

## Pasos para Descargar el Módulo de Detección Facial de SAMI

**Paso 1:**

Abre una terminal en el dispositivo a utilizar (Testeado en Raspberry Pi Modelo 4B) y escribe el siguiente comando:

```bash
git clone https://github.com/Andy1232866/SAMI.git
```

Esto descargará todos los archivos necesarios para la descarga del módulo de detección y conteo. Veremos 3 archivos, el Readme.md, el SAMI.py (script principal) y requirements.txt

**NOTA:** En caso de no tener git instalado lo puedes hacer con el siguiente comando:

```bash
sudo apt install git -y
```

**Paso 2:**

Antes de ejecutar el script deberás ejecutar los siguientes comandos en la misma termianl:

```bash
cd SAMI
pip install -r requirements.txt
```

Este comando instalará las dependencias necesarias para poder ejecutar el script sin problemas

**NOTA:** En caso de no tener python y pip instalado lo puedes hacer con el siguiente comando:

```bash
sudo apt update && sudo apt install python3 python3-pip -y
```

**NOTA:** Para comprobar si todas las dependencias se intalaron de manera correcta simplemente ejecuta la siguiente línea de comando:

```bash
python3 SAMI.py
```

En caso de que no se ejecute y de error de algún paquete es necesario instalarlo, esto se hace de la siguiente forma:

```bash
pip install [nombre del paquete]
```

**Paso 3:**

En el mismo directorio (/home/pi/SAMI - Habitualmente) escribir el siguiente comando:

```bash
git clone https://github.com/SaulCC23/pagina_SAMI.git
```

Para instalar las dependencias web necesarias escribirmos lo siguiente en la terminal:

```bash
cd pagina_SAMI/frontend
npm install
cd ../backend
npm install
```

**NOTA:** Es necesario esperar a que se instalen las dependencias de frontend para posteriormente viajar a la carpeta backend e instalar las dependencias necesarias
