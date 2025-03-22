# A grid-based environment with obstacles, rewards, and penalties.

# Project Structure
```
Projektas/
    ├── main.py               # Main entry point for the application
    ├── environment.py        # Environment implementation
    ├── agent.py              # Q-learning agent implementation
    ├── visualization.py      # Visualization components
    ├── utils.py              # Utility functions
    └── README.md             # Project documentation
```

# Projekto apžvalga
Šis projektas sukurtas kaip dirbtinio intelekto mokymosi užduotis, kurios tikslas - įgyvendinti pastiprinamojo mokymosi algoritmą realioje aplinkoje ir stebėti jo mokymosi procesą. Pagrindiniai komponentai:

- **Aplinka**: Tinklelio pasaulis su kliūtimis ir baudomis
- **Agentas**: Q-mokymosi algoritmas su epsilon-godžia veiksmo parinkimo strategija
- **Vizualizacija**: Realaus laiko animacija, parodanti agento mokymosi procesą

# 🚀 Įdiegimas
## Reikalavimai

- Python 3.7+
- NumPy
- Pygame
- Matplotlib
# Diegimo žingsniai
1. Klonuokite repozitoriją:
bash
```
git clone https://github.com/RubinoPD/Intelektika_projektas.git
```
2. Pereikite į projekto katalogą:
```
cd Intelektika_projektas
```
3. Įdiekite reikalingas bibliotekas:
```
pip install numpy pygame matplotlib
```
# 🎮 Naudojimas
Programą paleiskite komanda:
```
python main.py
```
# ⚙️ Parametrų konfigūracija
Programos parametrus galite keisti main.py faile:
```
# Aplinkos parametrai
env = GridWorld(
    width=10,               # Tinklelio plotis
    height=10,              # Tinklelio aukštis
    obstacle_density=0.2,   # Kliūčių tankis
    penalty_density=0.1,    # Baudų tankis
    max_steps=100           # Maksimalus žingsnių skaičius
)

# Agento parametrai
agent = QLearningAgent(
    num_states=env.get_num_states(),
    num_actions=env.get_num_actions(),
    learning_rate=0.1,      # Mokymosi koeficientas
    discount_factor=0.9,    # Nuolaidos faktorius
    exploration_rate=1.0,   # Pradinis tyrinėjimo koeficientas
    min_exploration_rate=0.01, # Minimalus tyrinėjimo koeficientas
    exploration_decay=0.995 # Tyrinėjimo koeficiento mažėjimo greitis
)
```
# 📊 Rezultatų analizė
Po mokymosi proceso galite analizuoti rezultatus naudodami išsaugotą Q-lentelę:
```
import numpy as np
import matplotlib.pyplot as plt

# Įkelti Q-table
q_table = np.load('q_table.npy')

# Vizualizuoti Q-vertes
plt.figure(figsize=(10, 8))
plt.imshow(q_table, cmap='viridis')
plt.colorbar(label='Q-value')
plt.title('Q-value matrix')
plt.xlabel('Actions')
plt.ylabel('States')
plt.show()
```
----
Projektas sukurtas kaip Intelektikos kurso darbas.