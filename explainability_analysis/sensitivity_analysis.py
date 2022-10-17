def calc_change_in_attn_weights(attn_weights_full_model, attn_weights_occluded_model, model_label):
    attn_weights_full_model = attn_weights_full_model.reset_index()
    attn_weights_occluded_model = attn_weights_occluded_model.reset_index()
    attn_weights_orig_and_occluded = attn_weights_full_model.merge(attn_weights_occluded_model, on=["index", "Date"],
                                                                   how="inner")
    attn_weights_orig_and_occluded["Attention Change"] = attn_weights_orig_and_occluded["Attention_y"] - \
                                                         attn_weights_orig_and_occluded["Attention_x"]

    attn_weights_orig_and_occluded = attn_weights_orig_and_occluded.rename(
        columns={"Crop type_y": "Crop type"})
    attn_weights_orig_and_occluded = attn_weights_orig_and_occluded.groupby(["Date", "Crop type"]).mean().reset_index()
    attn_weights_orig_and_occluded['model_label'] = model_label
    return attn_weights_orig_and_occluded