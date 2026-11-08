// Shared paper template (Typst 0.15).
#let paper(title: "", authors: (), abstract: [], keywords: (), body) = {
  set document(title: title, author: authors.map(a => a.name))
  set page(paper: "us-letter", margin: (x: 1.8cm, y: 2cm), numbering: "1", columns: 1)
  set text(font: ("Libertinus Serif", "New Computer Modern", "DejaVu Serif"), size: 10pt, lang: "en")
  set par(justify: true, leading: 0.58em, spacing: 0.9em)
  set heading(numbering: "1.1")
  show heading.where(level: 1): it => block(above: 1.3em, below: 0.7em, text(size: 12pt, weight: "bold", it))
  show heading.where(level: 2): it => block(above: 1.0em, below: 0.5em, text(size: 10.5pt, weight: "bold", it))
  show figure.caption: set text(size: 8.8pt)
  show table: set text(size: 8.4pt)
  show raw: set text(size: 8.6pt)
  show link: set text(fill: rgb("#1f4e9c"))

  align(center)[
    #block(text(size: 16pt, weight: "bold", title))
    #v(0.4em)
    #authors.map(a => [#a.name#if "affil" in a [ \ #text(size: 8.5pt, a.affil)]]).join(h(2em))
  ]
  v(0.8em)
  block(inset: (x: 1.2cm))[
    #set par(justify: true)
    #text(weight: "bold")[Abstract.] #abstract
    #if keywords.len() > 0 [ \ #text(size: 8.5pt)[*Keywords:* #keywords.join(", ")] ]
  ]
  v(0.6em)
  body
}

#let note(body) = block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%, text(size: 9pt, body))
