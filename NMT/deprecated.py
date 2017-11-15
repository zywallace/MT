def my_param(dict):
    dict["encoder.embeddings.weight"] = dict.pop("encoder.embeddings.emb_luts.0.weight")
    dict["decoder.embeddings.weight"] = dict.pop("decoder.embeddings.emb_luts.0.weight")
    dict["out.weight"] = dict.pop("0.weight")
    dict["out.bias"] = dict.pop("0.bias")
    for k,v in dict.items():
        if "layers.0" in k:
            dict[k.replace(".layers.0","") + "_l0"] = dict.pop(k)