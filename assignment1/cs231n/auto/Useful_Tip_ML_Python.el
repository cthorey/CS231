(TeX-add-style-hook
 "Useful_Tip_ML_Python"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "applemac") ("fontenc" "T1")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "inputenc"
    "amsmath"
    "graphicx"
    "fullpage"
    "fontenc"
    "color")
   (TeX-add-symbols
    "erf"
    "erfc")))

