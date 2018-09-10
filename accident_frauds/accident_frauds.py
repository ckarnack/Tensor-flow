
# coding: utf-8

# In[1]:


import pandas as pd


# Загрузка датасета

# In[2]:


data = pd.read_csv("D:/data.csv", encoding='cp1251', low_memory=False, delimiter=';')
list_1 = data['Участник 1'].tolist()
list_2 = data['Участник 2'].tolist()


# Представляем данные в виде графа. Вершины графа - участники ДТП, ребра указывают на факт наличия ДТП между участниками.
# Граф представляется в виде словаря. Ключ - вершина графа, значения - список соседних вершин.

# In[4]:


full_graph={}
for i, node in enumerate(list_1):
    if node not in full_graph.keys():
        full_graph[node] = [list_2[i]]
    else:
        full_graph[node].append(list_2[i])
        
    if list_2[i] not in full_graph.keys():
        full_graph[list_2[i]] = [node]
    else:
        full_graph[list_2[i]].append(node)


# Обозначим "вызывающими подозрение"  лиц, совершивших N_acc и более ДТП за рассматриваемый период времени
# Также будем искать группы лиц, которые совершали ДТП попарно (то есть образуют циклы в графе). Для этого выделяем из
# полного графа набор вершин, соответствующий людям, участвовавшим минимум в 2-х ДТП (набор подграфов исходного графа).

# In[5]:


subgraph = {}
N_acc = 3
print("Подозрения по количеству совершенных аварий (3 и более) вызывают следующие лица:")
for key in full_graph.keys():
    num_accidents = len(full_graph[key])
    if num_accidents>N_acc:
        print ("%s - участник %d ДТП" %(key, num_accidents))
    
    if num_accidents>1:
        subgraph[key]=full_graph[key]
    


# In[6]:


subgraph


# Представляем выделенный набор подграфов в виде списка ребер для удобства дальнейшего поиска.

# In[7]:


subgraph_edges = []
subgraph_nodes = set()
for item in subgraph.items():
    for i in item[1]:
        subgraph_edges.append([item[0],i])
        subgraph_nodes.add(item[0])
        subgraph_nodes.add(i)


# In[8]:


subgraph_edges, subgraph_nodes


# Опишем функцию поиска циклов в графе при старте из конкретной вершины. 

# In[10]:


def find_cycles(path, graph):
    start_node = path[0]
    next_node= None
    sub = []

#     В процессе прохода фиксируем пройденный путь как список вершин. 
#     Если на каком-то шаге следующая вершина эквивалентна той, из которой мы начали путь – значит есть цикл.
#     Вершины, образующие цикл представляют группу лиц-подозреваемых на мошеннические действия.
    
    for edge in graph:
        node1, node2 = edge
        if start_node in edge:
                if node1 == start_node:
                    next_node = node2
                else:
                    next_node = node1
        if next_node not in path:
                sub = [next_node]
                sub.extend(path)
                find_cycles(sub, graph);
        elif len(path) > 2  and next_node == path[-1]:
                # цикл найден, сохраняем       
                path.sort()
                path_string = ', '.join(path)
                if path_string not in cycles:
                    cycles.append(path_string)


# Запускаем процедуру поиска циклов для каждой вершины в subgraph. Результаты сохраняются в глобальную переменную cycles.

# In[11]:


cycles = []

global cycles

for node in subgraph_nodes:
    find_cycles([node],subgraph_edges)
print ("Подозрение вызывают указанные ниже группы лиц. Причина - организованная группа по совершению подставных ДТП:")
for i,cy in enumerate(cycles):
    print("  %d группа:  %s" %(i+1,cy))

