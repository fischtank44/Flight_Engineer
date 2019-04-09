
import sys
    
    
def export_linear_model_to_txt( file_name ):
    feat = [i for i in features.columns] 
    coef = [float(j) for j in model.coef_]
    out = []
    for fe, co in zip(feat, coef): 
    # print(fe, co)
        out.append([fe , co ])
    f = open(file_name, 'w')
    print(out)
    # f.close()
