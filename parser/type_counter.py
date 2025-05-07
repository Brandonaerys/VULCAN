from pathlib import Path




types = ['GasDwarf', 'Hycean','MiniNep']
mets = [30,50,75,100,125,150,175,200]
COs = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]

dfs = []
labels = []

for type in types:
    counter = 0
    for met in mets:
        for CO in COs:
            vul_data_name = f'../output/{type}_{int(met)}_{int(CO*100)}.vul'

            file_path = Path(vul_data_name)

            if file_path.exists():
                counter+=1

    print(f'{type} count: {counter}')
