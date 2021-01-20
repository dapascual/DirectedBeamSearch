# DirectedBeamSearch

This repository contains the code of the paper: "Directed Beam Search: Plug-and-Play Lexically Constrained LanguageGeneration", if you find this useful and use it for your own research, please cite us.

Currently under construction, more files an explanations will be added soon. The current files are enough to run the experiments, but ready-to-run examples will be added soon.

To run the method on 50 sets of 5 keywords do:

```
python encode_keywords.py

```
This will encode the keywords to glove for faster running. Then launch the main code as:


```
python main_DBS.py 
```
Inside this file some options might be changed, the default options run the model as in the experiments of the current version of the paper.

