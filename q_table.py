import numpy as np
import matplotlib.pyplot as plt

# Ikelti Q-table
q_table = np.load('q_table.npy')

# Isspausdinti lenteles forma ir kelias reiksmes
print("Q-table shape:", q_table.shape)
print("Sample values:\n", q_table[:5]) # Pirmos 5 Q-lenteles eilutes

# Vizualizuoti Q-vertes
plt.figure(figsize=(10, 8)) # Sukuriamas naujas langas nurodant jo dydi
plt.imshow(q_table, cmap='viridis') # imshow atvaizduoja 2D masyva, o viridis nustato spalvu palete
plt.colorbar(label='Q-value')
plt.title('Q-value matrix')
plt.xlabel('Actions')
plt.ylabel('State')
plt.show()

# Optimalių veiksmų vizualizacija
optimal_actions = np.argmax(q_table, axis=1)
action_names = ['Aukštyn', 'Dešinėn', 'Žemyn', 'Kairėn']

# Sukurkime tinklelio reprezentaciją (pvz., 10x10 tinkleliui)
grid_size = 10  # Keiskite pagal savo tinklelio dydį
optimal_actions_grid = optimal_actions.reshape(grid_size, grid_size)

plt.figure(figsize=(10, 8))
plt.imshow(optimal_actions_grid, cmap='tab10')
plt.colorbar(ticks=[0, 1, 2, 3], label='Optimal Action')
plt.title('Optimal Actions for Each State')

# Pridėkime veiksmo pavadinimus kiekvienai būsenai
for i in range(grid_size):
    for j in range(grid_size):
        action_idx = optimal_actions_grid[i, j]
        plt.text(j, i, action_names[action_idx],
                 ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
