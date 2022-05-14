from graphviz import Digraph

from datalabs import get_task, TaskType

"""Dynamic Generation of Task Typology Tree
By runing:
 python get_task_tree.py,
we can get a visualized version of task typology tree (`task-tree.png`) based on
the definition: https://github.com/ExpressAI/DataLab/tree/main/datalabs/tasks
"""


dot = Digraph("task-tree")

# setup node
for task_name in TaskType.list():
    dot.node(task_name, task_name)


# setup edge
edge_list = []
for task_name in TaskType.list():

    if task_name == "ROOT":
        continue

    task_categories = get_task(task_name)().task_categories
    for name in task_categories:
        if isinstance(name, TaskType):
            edge_list.append((name.value, task_name))
        else:
            edge_list.append(("ROOT", task_name))


# register edges
for v in edge_list:
    dot.edge(v[0], v[1])


# graph settings: see more details: https://graphviz.readthedocs.io/en/stable/index.html
u = dot.unflatten(stagger=8)  # max-depth
# format of output picture
u.format = "png"
print(u.source)
u.render(view=False)
