from graphviz import Digraph

dot = Digraph(comment='Transfer Learning', format='png')
dot.attr(rankdir='LR', dpi='300')
dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica')

# --- BLOCO 1: REDE PRÉ-TREINADA (CONGELADA) ---
with dot.subgraph(name='cluster_frozen') as c:
    c.attr(style='dashed', color='gray', label='CONHECIMENTO TRANSFERIDO (Congelado)')
    c.attr(fontcolor='gray')
    
    c.node('Input', 'Input Image\n(MRI 3D)', fillcolor='#E3F2FD')
    
    # Camadas Convolucionais
    c.node('L1', 'Conv Layers\n(Baixo Nível:\nBordas/Texturas)', fillcolor='#FFF9C4')
    c.node('L2', 'Dense Blocks\n(Médio Nível:\nFormas Anatômicas)', fillcolor='#FFF9C4')
    c.node('L3', 'Transition Layers\n(Alto Nível:\nPadrões Semânticos)', fillcolor='#FFF9C4')
    
    c.edge('Input', 'L1')
    c.edge('L1', 'L2')
    c.edge('L2', 'L3')

# --- BLOCO 2: VETOR DE CARACTERÍSTICAS ---
dot.node('Features', 'Vetor de Características\n(Embeddings 1024d)', shape='parallelogram', fillcolor='#C8E6C9')

# --- BLOCO 3: NOVOS CLASSIFICADORES (TREINÁVEIS) ---
with dot.subgraph(name='cluster_trainable') as c:
    c.attr(style='solid', color='black', label='NOVOS CLASSIFICADORES (Treináveis)')
    
    c.node('Ens', 'Ensemble Learning\n(SVM + RF + LR)', fillcolor='#FFCCBC')
    c.node('Out', 'Diagnóstico Final\n(Alzheimer/Controle)', shape='doubleoctagon', fillcolor='#E1BEE7')
    
    c.edge('Ens', 'Out')

# Conexões entre blocos
dot.edge('L3', 'Features', label=' Extração')
dot.edge('Features', 'Ens', label=' Input')

# Salvar
dot.render('transfer_learning_diagram', view=False)
print("Imagem 'transfer_learning_diagram.png' gerada!")