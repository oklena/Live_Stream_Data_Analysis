import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.ensemble import RandomForestRegressor

def MAPE(y_true, y_pred): 
    """
    Function takes true and predicted values
    and return Mean Absolute Percentage Error
    """    
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plot_feature_importances(model, data):
    """
    Function takes model name
    and plot feature importances of all the model's features 
    """
    n_features = data.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), data.columns.values) 
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')

    
def plot_top5_feature_importances(model, data, title):
    """
    Function takes model name
    and plot top 5 feature importances of the model 
    """
    if (type(model) == RandomForestRegressor):
        summary = list(zip(data.columns, model.feature_importances_))
        summary = sorted(summary, key=itemgetter(1), reverse=True)
        summary
    else:
        summary = list(zip(data.columns, model.coef_))
        summary = sorted(summary, key=itemgetter(1), reverse=True)
        summary
    
    ylabel = [i[0] for i in summary][:5]
    xlabel = [i[1] for i in summary][:5]
    fig, ax = plt.subplots()
    
    ax.barh(ylabel, xlabel, height=0.6, color=['#9146ff', '#b58eee'])
    ax.invert_yaxis()
    
    ax.axes.set_title(title, fontsize=20)
 

"""
Next function is adapted from an example code on scikit-learn: 
https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
"""
def plot_obs_pred(df, feature, weight, true, predicted, y_label=None,
                  title=None, ax=None, fill_legend=False):
    """ Plot observed and predicted - aggregated per feature level.

    Parameters
    ----------
    df : DataFrame
        input data
    feature: str
        a column name of df for the feature to be plotted
    weight : str
        column name of df with the values of weights or exposure
    observed : str
        a column name of df with the observed target
    predicted : DataFrame
        a dataframe, with the same index as df, with the predicted target
    fill_legend : bool, default=False
        whether to show fill_between legend
    """
    # aggregate observed and predicted variables by feature level
    df_ = df.loc[:, [feature, weight]].copy()
    df_["true"] = df[true] * df[weight]
    df_["predicted"] = predicted * df[weight]
    df_ = (
        df_.groupby([feature])[[weight, "true", "predicted"]]
        .sum()
        .assign(true=lambda x: x["true"] / x[weight])
        .assign(predicted=lambda x: x["predicted"] / x[weight])
    )

    ax = df_.loc[:, ["true", "predicted"]].plot(style=".", ax=ax, color=['#10061d', '#6c72e9'])
    y_max = df_.loc[:, ["true", "predicted"]].values.max() * 0.8
    p2 = ax.fill_between(
        df_.index,
        0,
        y_max * df_[weight] / df_[weight].values.max(),
        color='#9146ff',
        alpha=0.3,
    )
    if fill_legend:
        ax.legend([p2], ["{} distribution".format(feature)])
    ax.set(
        ylabel=y_label if y_label is not None else None,
        title=title if title is not None else "Train: True vs Predicted",
    )