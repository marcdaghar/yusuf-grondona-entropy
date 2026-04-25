
---

## Fichier 2 : `currency_aesthetics/generate_currency_art.py`

Créez un nouveau fichier :

1. Cliquez **"Add file"** → **"Create new file"**
2. Nom : `currency_aesthetics/generate_currency_art.py`
3. Copiez le code ci-dessous

```python
#!/usr/bin/env python3
"""
Génération d'images pour l'esthétique des supports monétaires

Produit:
- fibonacci_spiral.png : Spirale de Φ dans le rectangle d'or
- golden_rectangle.png : Construction du rectangle d'or
- transformed_banknote.png : Visualisation du billet comme lettre d'amour
- cymatics_intention.png : Motif cymatique de "Ce n'est pas de l'argent"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.path import Path
import matplotlib.patches as patches

# Configuration des polices pour un rendu esthétique
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11


def draw_fibonacci_spiral():
    """Dessine la spirale de Fibonacci dans le rectangle d'or"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Rectangle d'or
    phi = (1 + 5**0.5) / 2
    width = 10
    height = width / phi
    
    # Dessiner le rectangle
    rect = Rectangle((0, 0), width, height, fill=False, edgecolor='gold', linewidth=2)
    ax.add_patch(rect)
    
    # Points de la spirale
    theta = np.linspace(0, 4 * np.pi, 1000)
    r = np.exp(theta / (np.pi/2))
    x_spiral = r * np.cos(theta) * 0.3
    y_spiral = r * np.sin(theta) * 0.3
    ax.plot(x_spiral + width/2, y_spiral + height/2, 'red', linewidth=1.5)
    
    # Carrés de Fibonacci
    squares = [(0, 0, 1), (1, 1, 1), (2, 1, 2), (3, 2, 3), (4, 3, 5)]
    colors = ['#FFD700', '#FFC125', '#FFB90F', '#FFA500', '#FF8C00']
    
    x_offset, y_offset = 1, 1
    for i, (x, y, size) in enumerate(squares):
        square = Rectangle((x + x_offset, y + y_offset), size, size, 
                          fill=True, alpha=0.3, facecolor=colors[i % len(colors)], 
                          edgecolor='gold', linewidth=1)
        ax.add_patch(square)
    
    ax.set_xlim(-1, width + 2)
    ax.set_ylim(-1, height + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'La Spirale de Fibonacci\nRectangle d\'or (1:{phi:.3f})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('currency_aesthetics/fibonacci_spiral.png', dpi=150, bbox_inches='tight', 
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print("✅ fibonacci_spiral.png généré")


def draw_golden_rectangle():
    """Dessine la construction du rectangle d'or"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    phi = (1 + 5**0.5) / 2
    
    # Rectangle d'or
    width = 10
    height = width / phi
    
    # Fond
    ax.set_facecolor('#0d1117')
    
    # Rectangle principal
    rect = Rectangle((0, 0), width, height, fill=False, edgecolor='#FFD700', linewidth=2)
    ax.add_patch(rect)
    
    # Diagonale
    ax.plot([0, width], [0, height], 'r--', linewidth=1, alpha=0.7, label=f'Diagonale = √(1+Φ²) = {np.sqrt(1+phi**2):.3f}')
    
    # Ligne d'or
    golden_x = width / phi
    ax.axvline(x=golden_x, color='#00FF00', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Annotations
    ax.annotate(f'Φ = {phi:.4f}', xy=(golden_x/2, height/2), fontsize=12, color='#FFD700')
    ax.annotate('1', xy=(width - 1, -0.5), fontsize=10, color='white')
    ax.annotate(f'Φ = {phi:.3f}', xy=(width/2, height + 0.3), fontsize=10, color='white')
    
    ax.set_xlim(-1, width + 1)
    ax.set_ylim(-1, height + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Le Rectangle d\'Or\n', fontsize=14, color='white')
    
    plt.tight_layout()
    plt.savefig('currency_aesthetics/golden_rectangle.png', dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print("✅ golden_rectangle.png généré")


def draw_transformed_banknote():
    """Visualisation d'un billet transformé en lettre d'amour"""
    fig, ax = plt.subplots(figsize=(14, 9))
    
    ax.set_facecolor('#f5f0e1')  # Papier ancien
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    
    # Billet (rectangle aux proportions 2:1)
    banknote = FancyBboxPatch((2, 2), 10, 5, boxstyle="round,pad=0.1",
                               facecolor='#e8e4d8', edgecolor='#8B7355', linewidth=2)
    ax.add_patch(banknote)
    
    # Motifs du billet (transformés)
    # Centre : cœur φ-optimal
    heart = patches.RegularPolygon((7, 4.5), 5, radius=0.8, orientation=np.pi/5,
                                    facecolor='#FF6B6B', alpha=0.7)
    ax.add_patch(heart)
    
    # Écriture manuscrite simulée
    text_lines = [
        "Ce billet ne sert à rien",
        "dans le monde marchand.",
        "",
        "Je te donne ce papier",
        "parce que je pense à toi.",
        "",
        "Il n'est pas pour acheter.",
        "Il est pour dire :",
        "tu existes."
    ]
    
    y_start = 5.8
    for i, line in enumerate(text_lines):
        ax.text(4, y_start - i * 0.4, line, fontsize=9, fontfamily='cursive',
                color='#2c1810', alpha=0.9)
    
    # Nombre d'or en filigrane
    phi = (1 + 5**0.5) / 2
    ax.text(11, 2.5, f'Φ = {phi:.4f}', fontsize=7, alpha=0.3, color='#8B7355',
            rotation=90)
    
    # Éléments décoratifs
    ax.plot([2, 12], [4, 4], 'r-', linewidth=0.5, alpha=0.3)
    ax.plot([2, 12], [5, 5], 'r-', linewidth=0.5, alpha=0.3)
    
    ax.axis('off')
    ax.set_title('Le Billet Transformé — Lettre d\'Amour', fontsize=14, color='#2c1810')
    
    plt.tight_layout()
    plt.savefig('currency_aesthetics/transformed_banknote.png', dpi=150, bbox_inches='tight',
                facecolor='#f5f0e1', edgecolor='none')
    plt.close()
    print("✅ transformed_banknote.png généré")


def draw_cymatics():
    """Génère un motif cymatique pour 'Ce n'est pas de l'argent'"""
    # Fréquences correspondant aux voyelles de la phrase
    # Approximation des formants : a(850Hz), e(500Hz), i(350Hz), etc.
    frequencies = [350, 500, 850, 280, 720, 420, 620]  # 'c e n' e s t p a s'
    harmonics = [0.5, 1, 1.5, 2, 2.5, 3]
    
    # Grille
    x = np.linspace(-1, 1, 500)
    y = np.linspace(-1, 1, 500)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    
    # Superposition des ondes stationnaires
    for freq in frequencies:
        k = freq / 1000  # Facteur d'échelle
        for harm in harmonics:
            Z += np.sin(k * harm * np.pi * X) * np.cos(k * harm * np.pi * Y)
    
    # Normalisation
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Tracé avec colormap vibrante
    im = ax.imshow(Z, cmap='magma', origin='lower', extent=[-1, 1, -1, 1])
    
    # Ajout du texte
    ax.text(0, 0, 'Ce n\'est\npas\nde l\'argent', 
            fontsize=18, fontweight='bold', color='white',
            ha='center', va='center', alpha=0.9)
    
    ax.axis('off')
    ax.set_title('Cymatique de l\'Intention\nLa vibration de "Ce n\'est pas de l\'argent"', 
                 fontsize=12, color='white')
    
    plt.tight_layout()
    plt.savefig('currency_aesthetics/cymatics_intention.png', dpi=150, bbox_inches='tight',
                facecolor='#0d0d0d', edgecolor='none')
    plt.close()
    print("✅ cymatics_intention.png généré")


def main():
    """Génère toutes les images esthétiques"""
    print("\n" + "=" * 60)
    print(" GÉNÉRATION DES IMAGES ESTHÉTIQUES")
    print(" Violence monétaire → Lettre d'amour")
    print("=" * 60 + "\n")
    
    draw_fibonacci_spiral()
    draw_golden_rectangle()
    draw_transformed_banknote()
    draw_cymatics()
    
    print("\n" + "=" * 60)
    print("✅ 4 images générées dans currency_aesthetics/")
    print("   - fibonacci_spiral.png")
    print("   - golden_rectangle.png")
    print("   - transformed_banknote.png")
    print("   - cymatics_intention.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
