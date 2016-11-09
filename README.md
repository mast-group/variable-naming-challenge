# The Java Variable Naming Challenge

The goal of the challenge is to automatically predict the name of a variable given its context.
Learning to name variables is an interesting problem in the intersection of machine learning,
software engineering and natural language processing since it requires some level of 
understanding of source code functionality as well as linguistic information. The challenge
has important links to coding conventions and code comprehension.

A detailed description of the challenge, the data and the evaluation metrics can be found 
in the relevant technical report [here](todo).

#### Code
This repository contains the `naturalize` Python package (compatible with Python 2.7 and 3.5) for
evaluating naming models. To use the code, implement the `AbstractRenamingModel` and use the
`point_suggestion_eval.py` to evaluate your model.


#### Data
The challenge data can be accessed from [here](http://groups.inf.ed.ac.uk/cup/var-naming-challenge/data.zip).
These include the original Java files as well as parsed JSON files.

#### Related Work
Work related to the variable naming challenge:
* Allamanis, Barr, Bird, Sutton. "Learning natural coding conventions", FSE 2014. [[link]](http://groups.inf.ed.ac.uk/naturalize/)
* Allamanis, Barr, Bird, Sutton. "Suggesting accurate method and class names", FSE 2015. [[link]](http://groups.inf.ed.ac.uk/cup/naturalize/)
* Raychev, Vechev, Krause. "Predicting program properties from Big Code", PoPL 2015. [[link]](http://www.srl.inf.ethz.ch/jsnice)
* Allamanis, Peng, Sutton "A convolutional attention network for extreme summarization of source code" ICML 2016. [[link]](http://groups.inf.ed.ac.uk/cup/codeattention/)