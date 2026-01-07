# Paper on Centralities and Normalisation

Paper on centralities and normalisation methods

## Installation

Clone this repository to a local working folder.

A python package manager and an IDE such as `vscode` are recommended.

### UV

The UV package manager can be installed on mac per `brew install uv`. Packages can then be installed into a virtual environment per `uv sync`.

### IDE and venv

The virtual environment should be detected automatically by IDEs such as vscode, else activate it manually.

## Urban Data

The main dataset is prepared by running [Madrid UA Dataset](https://github.com/songololo/ua-dataset-madrid).

Generate a copy of the dataset and copy `dataset.gpkg` to this repository in a folder called `temp`. The `temp` folder is ignored per `.gitignore` but is required for the dataset to be found by the Python scripts.

## Travel Survey Data

- [Travel Survey](https://datos.crtm.es/documents/6afd4db8175d4902ada0803f08ccf50e/about)
- [Geographic Zones](https://datos.crtm.es/datasets/crtm::zonificacionzt1259/about)
- [License](https://www.crtm.es/licencia-de-uso)
  - Static data license from the Regional Transport Consortium of Madrid for the open data portal of the CRTM website
  - Powered by CRTM
  - https://www.crtm.es/licencia-de-uso
- See below for translation of travel survey fields.

```
@misc{crtm_edm_2020,
  title        = {{Encuesta Domiciliaria de Movilidad (EDM) 2018}},
  author       = {{Consorcio Regional de Transportes de Madrid}},
  year         = {2020},
  month        = feb,
  howpublished = {Microsoft Excel Spreadsheet},
  url          = {https://datos.crtm.es/documents/6afd4db8175d4902ada0803f08ccf50e/about},
  note         = {Travel survey data for the Madrid metropolitan region. File size: 22.88 MB. Published: 27 February 2020. Open data license from the Madrid Regional Transport Consortium, allowing commercial and non-commercial reuse with attribution}
}

@misc{crtm_zones_2020,
  title        = {{ZonificacionZT1259}},
  author       = {{Consorcio Regional de Transportes de Madrid}},
  year         = {2020},
  month        = feb,
  howpublished = {Feature Layer Dataset},
  url          = {https://datos.crtm.es/search?q=1259},
  note         = {Transport zones (ZT1259) from EDM2018. 1,259 geographic zones for the Community of Madrid at a territorial scale between neighborhood and census section. Published: 29 August 2019. Updated: 28 February 2020. Open data license from the Madrid Regional Transport Consortium, allowing commercial and non-commercial reuse with attribution}
}

@misc{madrid_premises,
  title        = {Census of Premises and Activities of the Madrid City Council},
  author       = {{Madrid City Council}},
  year         = {2014},
  howpublished = {Dataset},
  url          = {https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=66665cde99be2410VgnVCM1000000b205a0aRCRD},
  note         = {Open data. Origin of the data: Madrid City Council. Licensed under Spanish Law 37/2007 on Reuse of Public Sector Information. License terms: \url{https://datos.madrid.es/egob/catalogo/aviso-legal}}
}

@misc{madrid_street_network,
  title        = {Callejero de la Comunidad de Madrid},
  author       = {{Community of Madrid}},
  year         = {2019},
  howpublished = {Dataset},
  url          = {https://datos.comunidad.madrid/catalogo/dataset/spacm_callescm},
  note         = {Open data. Set of roads officially approved by the municipalities of the Community of Madrid. Licensed under Creative Commons Attribution 4.0 (CC BY 4.0). License terms: \url{https://creativecommons.org/licenses/by/4.0/legalcode.es}. Original dataset link since removed: \url{https://datos.comunidad.madrid/catalogo/dataset/spacm_callescm} Possible replacements from \url{https://data.europa.eu/data/datasets/https-idem-madrid-org-catalogocartografia-srv-resources-datasets-spacm_callescm?locale=en} \url{https://gestiona.comunidad.madrid/iestadis/fijas/estructu/general/territorio/estructu_descargas.htm} \url{https://gestiona.comunidad.madrid/nomecalles_web/#/inicio} via Download Calejero.}
}
```

## Processing

Run the plot scripts in the `scripts` folder. It is recommended to use an IDE such as vscode to run the cell blocks directly. Cell blocks are used instead of Jupyter notebooks because the latter can cause complications and bloat for code repositories.

## Paper Compilation

To compile the LaTeX paper:

```bash
cd paper
latexmk main.tex
```

Create a submission package:

```bash
cd paper
# Copy all images to a flat structure
mkdir -p submission_files
```

To generate a single combined LaTeX file for journal submission (e.g., Editorial Manager):

```bash
cd paper
latexpand --makeatletter main.tex > submission_files/main_combined.tex
```

This expands all `\input` commands into a single file. The `--makeatletter` flag handles internal LaTeX commands that use the `@` symbol (like `\c@figure`).

```bash
cp main.bbl submission_files/
cp arxiv.sty submission_files/
cp images/*.pdf submission_files/
cp plots/*.pdf submission_files/
cp plots/*.png submission_files/
```

**Note:** Editorial Manager requires a flat structure without subdirectories. The image paths in `main_combined.tex` reference `images/` and `plots/` directories, so you'll need to either:

1. Remove the directory prefixes from all `\includegraphics` paths in `main_combined.tex`, or
2. Upload files individually through Editorial Manager's web interface (recommended)

For option 1, run:

```bash
sed -i.bak 's|{images/|{|g; s|{plots/|{|g' submission_files/main_combined.tex
```

ZIP the `submission_files` folder for upload.

```bash
# Create flat zip without directory structure
cd submission_files
zip ../submission.zip *
cd ..
```

## Travel Survey Fields

- **ID_HOGAR** - Household Identifier
- **ID_IND** - Individual Identifier
- **ID_VIAJE** - Trip Identifier
- **VORI** - Origin. Reason for departure
  1. Home
  2. Work
  3. Work management
  4. Study
  5. Shopping
  6. Medical
  7. Accompanying another person
  8. Leisure
  9. Sport / take a walk
  10. Personal matter
  11. Another residence
  12. Others
- **VORIHORAINI** - Origin. Start time
- **VDES** - Destination. Reason for destination
  1. Home
  2. Work
  3. Work management
  4. Study
  5. Shopping
  6. Medical
  7. Accompanying another person
  8. Leisure
  9. Sport / take a walk
  10. Personal matter
  11. Another residence
  12. Others
- **VDESHORAFIN** - Destination. Arrival time
- **VFRECUENCIA** - Trip frequency
  1. Daily, Monday to Friday
  2. Between 2 and 4 working days per week
  3. Less than two working days per week
  4. Occasionally
  5. It's the first time I make this trip
- **VVEHICULO** - Do you have a private vehicle for this trip?
  1. Yes
  2. No
- **VNOPRIVADO** - Why didn't you use the car as a driver throughout your trip?
  1. It's more expensive
  2. Parking difficulties
  3. It takes longer
  4. To avoid traffic jams
  5. More uncomfortable
  6. I don't like the car
  7. I pollute less
  8. Others
- **VNOPUBLICO** - Why didn't you use public transport on this trip (any of its stages)?
  1. Poor public transport combination
  2. No public service available
  3. Due to lack of knowledge/information
  4. I need my vehicle for work or personal management
  5. It takes longer
  6. It's more expensive
  7. More uncomfortable
  8. I don't like public transport
  9. My personal situation influences this modal choice
  10. The destination is very close
  11. I prefer walking / cycling
  12. Others
- **VORIZT1259** - Origin. Zone 1259
- **VDESZT1259** - Destination. Zone 1259
- **TIPO_ENCUESTA** - Type of survey
- **N_ETAPAS_POR_VIAJE** - Number of stages per trip
- **MOTIVO_PRIORITARIO** - Main reason for the trip (same list as "Origin. Reason for departure" and "Destination. Reason for destination")
- **DISTANCIA_VIAJE** - Distance, in kilometers, from origin to destination of the trip
- **MODO_PRIORITARIO** - Primary mode of travel
  1. Renfe Cercanías (Commuter Train)
  2. Interurban Bus
  3. Urban Bus from another municipality
  4. Metro
  5. Light Metro/Tram
  6. Urban Bus Madrid EMT
  7. Other Renfe services
  8. Discretionary Bus
  9. Long Distance Bus
  10. Taxi
  11. Private car (driver)
  12. Company car (driver)
  13. Rental car without driver
  14. Private car (passenger)
  15. Company car (passenger)
  16. Rental car with driver
  17. Private Motorcycle/Scooter
  18. Public Motorcycle/Scooter
  19. Company Motorcycle/Scooter
  20. Private Bicycle
  21. Public Bicycle
  22. Company Bicycle
  23. Others
  24. Walking
- **ELE_G_POND_ESC2** - Travel Elevator
