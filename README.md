# Paper on Centralities and Normalisation

Paper on centralities and normalisation methods

## Installation

Clone this repository to a local working folder.

A python package manager and an IDE such as `vscode` are recommended.

### UV

The UV package manager can be installed on mac per `brew install uv`. Packages can then be installed into a virtual environment per `uv sync`.

### PDM

The PDM package manager can be installed on mac per `brew install pdm`. Packages can then be installed into a virtual environment per `pdm install`.

### IDE and venv

If using an IDE the `.venv` should be detected automatically by IDEs such as vscode, else activate it manually.

## Urban Data

The main dataset is prepared by running the [Madrid UA Dataset](https://github.com/songololo/ua-dataset-madrid).

Generate a copy of the dataset and copy `dataset.gpkg` to this repository in a folder called `temp`. The `temp` folder is ignored per `.gitignore` but is required for the dataset to be found by the Python scripts.

## Travel Survey Data

- [Travel Survey](https://datos.crtm.es/documents/6afd4db8175d4902ada0803f08ccf50e/about)
- [Geographic Zones](https://datos.crtm.es/search?q=1259)
- [License](https://datos.madrid.es/egob/catalogo/aviso-legal)
  - Static data license from the Regional Transport Consortium of Madrid for the open data portal of the CRTM website
  - Powered by CRTM
  - http://www.crtm.es/
- See below for translation of travel survey fields.

## Processing

It is recommended to use an IDE such as vscode to run the cell blocks directly. Cell blocks are used instead of Jupyter notebooks because the latter can cause issues and bloat for code repositories.

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
