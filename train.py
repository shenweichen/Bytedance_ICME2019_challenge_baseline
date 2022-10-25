import pandas as pd
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score

from model2 import xDeepFM_MTL

ONLINE_FLAG = False
loss_weights = [1, 1, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例

if __name__ == "__main__":
    data = pd.read_csv('./input/final_track2_sample.txt', sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did',
        'creat_time', 'video_duration'])
    if ONLINE_FLAG:
        test_data = pd.read_csv('./input/final_track2_test_no_anwser.txt', sep='\t', names=[
            'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did',
            'creat_time', 'video_duration'])
        train_size = data.shape[0]
        data = data.append(test_data)
    else:
        train_size = int(data.shape[0] * (1 - VALIDATION_FRAC))

    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did', ]
    dense_features = ['video_duration']  # 'creat_time',

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    target = ['finish', 'like']

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    sparse_feature_list = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=8)
                           for feat in sparse_features]
    dense_feature_list = [DenseFeat(feat, 1)
                          for feat in dense_features]

    feature_columns = sparse_feature_list + dense_feature_list
    feature_names = get_feature_names(feature_columns)

    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    train_labels = [train[target[0]].values, train[target[1]].values]
    test_labels = [test[target[0]].values, test[target[1]].values]

    model = xDeepFM_MTL(feature_columns)
    model.compile("adagrad", "binary_crossentropy", loss_weights=loss_weights, )

    if ONLINE_FLAG:
        history = model.fit(train_model_input, train_labels,
                            batch_size=4096, epochs=1, verbose=1)
        pred_ans = model.predict(test_model_input, batch_size=2 ** 14)

    else:
        history = model.fit(train_model_input, train_labels,
                            batch_size=4096, epochs=1, verbose=1)
        pred_ans = model.predict(test_model_input, batch_size=2 ** 14)

        print("finish AUC", round(roc_auc_score(test_labels[0], pred_ans[0]), 4))
        print("finish LogLoss", round(log_loss(test_labels[0], pred_ans[0]), 4))

        print("like AUC", round(roc_auc_score(test_labels[1], pred_ans[1]), 4))
        print("like LogLoss", round(log_loss(test_labels[1], pred_ans[1]), 4))

    if ONLINE_FLAG:
        result = test_data[['uid', 'item_id', 'finish', 'like']].copy()
        result.rename(columns={'finish': 'finish_probability',
                               'like': 'like_probability'}, inplace=True)
        result['finish_probability'] = pred_ans[0]
        result['like_probability'] = pred_ans[1]
        result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv(
            'result.csv', index=None, float_format='%.6f')
