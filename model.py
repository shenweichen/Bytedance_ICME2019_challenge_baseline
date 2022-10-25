import tensorflow as tf
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.interaction import CIN
from deepctr.layers.utils import concat_func, add_func

from deepctr.feature_column import build_input_features, get_linear_logit, input_from_feature_columns


def xDeepFM_MTL(feature_columns, hidden_size=(256, 256), cin_layer_size=(256, 256),
                cin_split_half=True,
                task_net_size=(128,), l2_reg_linear=0.00001, l2_reg_embedding=0.00001,
                seed=1024, ):
    if len(task_net_size) < 1:
        raise ValueError('task_net_size must be at least one layer')

    # video_input = tf.keras.layers.Input((128,))
    # inputs_list.append(video_input)

    features = build_input_features(feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(features, feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    deep_emb_list, dense_value_list = input_from_feature_columns(features, feature_columns,
                                                                 l2_reg_embedding, seed)

    fm_input = concat_func(deep_emb_list, axis=1)

    deep_input = tf.keras.layers.Flatten()(fm_input)
    deep_out = DNN(hidden_size)(deep_input)

    finish_out = DNN(task_net_size)(deep_out)
    finish_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(finish_out)

    like_out = DNN(task_net_size)(deep_out)
    like_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(like_out)

    finish_logit = add_func(
        [linear_logit, finish_logit])
    like_logit = add_func(
        [linear_logit, like_logit])

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, 'relu',
                       cin_split_half, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, activation=None, )(exFM_out)
        finish_logit = add_func([finish_logit, exFM_logit])
        like_logit = add_func([like_logit, exFM_logit])

    output_finish = PredictionLayer('binary', name='finish')(finish_logit)
    output_like = PredictionLayer('binary', name='like')(like_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=[
        output_finish, output_like])
    return model
