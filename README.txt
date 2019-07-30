Cross identification of radio astronomy objects using machine learning
James Gardner, Cheng Soon Ong, Matthew Alger
README

This project considers the problem of cross identification of objects in radio astronomy. Cross identification is the task of matching objects in one sky survey to the corresponding object in a different sky survey. We implement positional matching and machine learning based binary classification for solving the task of cross id. The methods are applied to NVSS (NRAO VLA Sky Survey) and TGSS (TIFR GMRT Sky Survey Alternative Data Release 1).

positionalmatching.ipynb takes two lists of celestial co-ordinates and creates a mapping for each into the other, taking the nearest point within a accepted distance.
