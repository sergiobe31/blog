# Blog TFM - Rebalanceador de Estrategias

Blog personal para documentar el desarrollo del Trabajo Fin de Máster.

## Configuración inicial

### 1. Instalar Ruby y Jekyll

**Windows (recomendado usar RubyInstaller):**
1. Descarga RubyInstaller con DevKit desde: https://rubyinstaller.org/
2. Ejecuta el instalador y selecciona "Add Ruby to PATH"
3. Al finalizar, ejecuta `ridk install` (opción 3)

**Verificar instalación:**
```bash
ruby -v
gem -v
```

### 2. Instalar Jekyll

```bash
gem install jekyll bundler
```

### 3. Instalar dependencias del blog

```bash
cd blog
bundle install
```

### 4. Ejecutar en local

```bash
bundle exec jekyll serve
```

Abre http://localhost:4000 en tu navegador.

## Crear nuevos posts

1. Crea un archivo en `_posts/` con el formato: `YYYY-MM-DD-titulo-del-post.md`
2. Añade el front matter al inicio:

```yaml
---
layout: post
title: "Título de tu post"
date: 2025-01-12
categories: [categoria1, categoria2]
---
```

3. Escribe tu contenido en Markdown

## Añadir imágenes

1. Coloca las imágenes en `assets/images/`
2. Referéncialas en tu post:

```markdown
![Descripción de la imagen](/assets/images/nombre-imagen.png)
```

## Fórmulas matemáticas

Usa LaTeX entre `\\(` y `\\)` para fórmulas inline, o `$$` para bloques:

```markdown
La fórmula inline \\(E = mc^2\\) se ve así.

Bloque de fórmula:
$$\hat{\beta} = (X'X)^{-1}X'y$$
```

## Código

Usa triple backtick con el lenguaje:

````markdown
```python
import pandas as pd
df = pd.read_csv('data.csv')
```
````

## Publicar en GitHub Pages

### 1. Crear repositorio en GitHub

- Ve a github.com y crea un nuevo repositorio
- Nombre sugerido: `blog` o `tfm-blog`

### 2. Subir el código

```bash
cd blog
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/TU-REPO.git
git push -u origin main
```

### 3. Activar GitHub Pages

1. Ve a Settings > Pages en tu repositorio
2. Source: selecciona "Deploy from a branch"
3. Branch: selecciona `main` y `/ (root)`
4. Guarda

### 4. Actualizar configuración

Edita `_config.yml` y actualiza:
```yaml
url: "https://tu-usuario.github.io"
baseurl: "/nombre-repo"
```

Tu blog estará disponible en: `https://tu-usuario.github.io/nombre-repo`

## Estructura del proyecto

```
blog/
├── _config.yml          # Configuración del sitio
├── _layouts/            # Plantillas HTML
│   ├── default.html
│   ├── post.html
│   └── page.html
├── _posts/              # Tus posts (archivos .md)
├── assets/
│   ├── css/
│   │   └── main.css     # Estilos
│   └── images/          # Imágenes
├── about.md             # Página "Sobre mí"
├── index.html           # Página principal
└── README.md            # Este archivo
```

## Tips

- **Preview local**: Siempre prueba con `bundle exec jekyll serve` antes de publicar
- **Drafts**: Crea posts en `_drafts/` sin fecha para trabajar en borradores
- **SEO**: Completa bien el `title` y `description` en cada post
