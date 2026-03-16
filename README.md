# geo-joy.github.io

Personal site of **Geo Joy** — Senior Architect, AI team lead, builder turned breaker.

Live at: [geo-joy.github.io](https://geo-joy.github.io)

---

## Design

A heavily customised Jekyll site built on top of [Academic Pages](https://github.com/academicpages/academicpages.github.io), redesigned from scratch with a terminal/OS aesthetic:

- **Fixed sidebar** — terminal card with macOS-style dots (`whoami`), avatar, bio, social icons, navigation, and footer
- **No header** — navigation lives entirely in the sidebar
- **Windowed blog** — blog posts open as draggable, resizable, minimisable floating windows over the blog list
- **Malayalam matrix rain** — animated background using Malayalam script characters
- **Frosted glass panels** — content areas with backdrop blur
- **Terminal section headers** — `// origin`, `// build`, `// now` style section markers
- **Dark navy theme** — `#0a1628` background, teal (`#2dd4bf`) + orange (`#f4860a`) accent palette
- **Mobile** — off-canvas sidebar with hamburger toggle

---

## Stack

- [Jekyll](https://jekyllrb.com/) — static site generator
- [GitHub Pages](https://pages.github.com/) — hosting
- Vanilla JS — window manager, matrix rain, mobile menu
- SCSS — custom theme, no external CSS frameworks

---

## Structure

```
_config.yml           # site config, author info, SEO
_pages/
  about.md            # landing page
  year-archive.html   # blog list (windowed posts)
  portfolio.html      # projects (live GitHub repos via API)
_posts/               # blog posts
_layouts/
  default.html        # base layout (sidebar injected here)
  single.html         # blog post layout
  archive.html        # list pages
_includes/
  author-profile.html # sidebar card
  sidebar.html        # sidebar wrapper
_sass/
  _custom.scss        # all custom styles
  layout/_sidebar.scss
  layout/_page.scss
  theme/_default_dark.scss
```

---

## Running locally

```bash
bundle install
bundle exec jekyll serve
```

Open `http://localhost:4000`

---

## Content

- **Blog** — `/blog/` — posts in `_posts/`
- **Projects** — `/portfolio/` — live GitHub repos fetched via API
- **Videos** — links to [YouTube/@breachguru](https://www.youtube.com/@breachguru)
- **CV** — links to [LinkedIn](https://linkedin.com/in/-itsg)

---

*Forked from [Academic Pages](https://github.com/academicpages/academicpages.github.io). Original theme by Michael Rose ([Minimal Mistakes](https://mademistakes.com/work/minimal-mistakes-jekyll-theme/)).*
