import matplotlib.pyplot as plt


def plot_measurements(width_data_mm):
    colors = ['b', 'g', 'r']
    for i in range(1, 4):
        if width_data_mm[i]:
            plt.figure(figsize=(8, 5))
            plt.plot(range(len(width_data_mm[i])), width_data_mm[i],
                    label=f'Метка {i}', color=colors[i-1])
            plt.xlabel('Время (кадры)')
            plt.ylabel('Ширина туловища (мм)')
            plt.title(f'Изменение ширины туловища для метки {i}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()