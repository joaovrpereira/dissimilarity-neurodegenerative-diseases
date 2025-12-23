from graphviz import Digraph

# Configuração do Diagrama
dot = Digraph(comment='Fluxo do Projeto', format='png')
dot.attr(rankdir='TB') # Top to Bottom (De cima para baixo)
dot.attr(dpi='300')    # Alta resolução

# --- NÓS (As caixinhas) ---

# 1. Entrada de Dados
dot.node('A', 'OASIS-3 Dataset\n(MRI + Clínico)', shape='cylinder', style='filled', fillcolor='#E0E0E0')

# 2. Engenharia de Dados
dot.node('B', 'Engenharia de Dados\n(Pareamento Temporal + Snapshot Único)', shape='box', style='filled,rounded', fillcolor='#B3E5FC')

# 3. Pré-processamento
dot.node('C', 'Pré-processamento 3D (MONAI)\n(Reamostragem 1mm, Crop 96x96x96)', shape='box', style='filled', fillcolor='#B3E5FC')

# 4. Deep Learning (Extrator)
dot.node('D', 'Feature Extraction\n(DenseNet121 3D - Pré-treinada)', shape='component', style='filled', fillcolor='#FFCC80')

# 5. Vetor Latente
dot.node('E', 'Vetor de Características\n(Deep Features - 1024d)', shape='parallelogram', style='filled', fillcolor='#FFF9C4')

# 6. Ensemble (Junta Médica)
dot.node('F', 'Ensemble Learning\n(Votação Suave)', shape='box3d', style='filled', fillcolor='#C8E6C9')

# Os 3 modelos dentro do ensemble (apenas indicativo)
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node('M1', 'SVM', shape='ellipse', style='filled', fillcolor='white')
    s.node('M2', 'Random Forest', shape='ellipse', style='filled', fillcolor='white')
    s.node('M3', 'Log. Regression', shape='ellipse', style='filled', fillcolor='white')

# 7. Decisão
dot.node('G', 'Threshold Moving\n(Limiar Otimizado = 0.20)', shape='diamond', style='filled', fillcolor='#F8BBD0')

# 8. Saída
dot.node('H', 'Diagnóstico Final\n(Alzheimer vs Controle)', shape='doubleoctagon', style='filled', fillcolor='#E1BEE7')

# --- ARESTAS (As setas) ---
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')

# Conecta vetor aos modelos
dot.edge('E', 'M1')
dot.edge('E', 'M2')
dot.edge('E', 'M3')

# Conecta modelos ao Ensemble
dot.edge('M1', 'F')
dot.edge('M2', 'F')
dot.edge('M3', 'F')

dot.edge('F', 'G', label=' Probabilidade Média')
dot.edge('G', 'H', label=' Decisão Final')

# Salvar e Renderizar
output_path = 'fluxo_projeto_completo'
dot.render(output_path, view=False)
print(f"Diagrama salvo como {output_path}.png")