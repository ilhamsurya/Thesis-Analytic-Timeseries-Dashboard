![Python](https://img.shields.io/badge/Python-^3.8-blue.svg?logo=python&longCache=true&logoColor=white&colorB=5e81ac&style=flat-square&colorA=4c566a)
![Flask](https://img.shields.io/badge/Flask-1.1.2-blue.svg?longCache=true&logo=flask&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![Flask-Assets](https://img.shields.io/badge/Flask--Assets-v2.0-blue.svg?longCache=true&logo=flask&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![Pandas](https://img.shields.io/badge/Pandas-v^1.0.0-blue.svg?longCache=true&logo=python&longCache=true&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![Dash](https://img.shields.io/badge/Dash-v1.12.0-blue.svg?longCache=true&logo=python&longCache=true&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)
![Plotly](https://img.shields.io/badge/Plotly-v4.8.1-blue.svg?longCache=true&logo=python&longCache=true&style=flat-square&logoColor=white&colorB=5e81ac&colorA=4c566a)

## Flask Boilerplate Modif V.1.0 [LastUpdate]: 28/04/2021

## Packages

Dillinger uses a number of open source projects to work properly:

- [Flask] - Engine Backend
- [TailwindCSS] - Library Frontend
- [Pandas] - Library buat pengolahan data
- [mysql-connector-python] - Buat konek ke mySQL
- [dash] - Framework visualisasi grafik
- [numpy] - Pengolahan data numerik
- [plotly] - Plotiing grafik,
- [gunicorn] - Web Server Gateway Interface

## Cara Instalasi Env Flask

1. `pip install pipenv ` install pipenv dulu
2. `pipenv shell ` mulai virtual environmentnya
3. `pipenv install ` Buat install dependency
4. `pipenv sync ` Buat install dependency (kalo abis nge list ga muncul)
5. `pipenv list ` Periksa dependency udah terinstall atau belum
6. `pipenv lock ` Generate file pipfile.lock
7. `flask run ` Jalanin projek flask nya

- **Dokumentasi buat pipenv**: https://pypi.org/project/pipenv/

## Cara Instalasi Tailwind CSS

1. `npm install ` untuk membuat node modules berdasarkan package.json
2. `npm run develop:css ` Generate front-end nya sebelum backend `flask run `
3. `npm run build:css` Buat ngubah state ke production

- **Dokumentasi buat tailwindCSS**: https://tailwindcss.com/docs

## Cara Instalasi Env Docker

### pastikan docker udah nyala

1. `docker build -t dashboardta ` create image baru buat projek ini
2. `docker run -p 5000:5000 dashboardta ` menjalankan docker
3. `http://127.0.0.1:5000/ ` Access Point

## Penjelasan mengenai boilerplate

1. Folder `App -> Static & Templates`: Buat kebutuhan frontend
2. Folder `Timeseries`: Buat logik pengolahan data time series
3. Folder `Dataset`: Dataset pelanggaran laut

Credit:

- **Flask-Docker**: https://gitlab.com/dfederschmidt/docker-pipenv-sample/container_registry
- **TailwindCSS**: https://github.com/blackbaba/Flask-Tailwind-Starter
