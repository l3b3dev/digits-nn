from Plotter import Plotter
from TrainingPipeline import TrainingPipeline
from sklearn import model_selection
import pandas as pd

if __name__ == '__main__':
    data_dir = "data"

    pipeline = TrainingPipeline()
    image_datasets, loaders = pipeline.initialize_data(data_dir)
    X_train, Y_train, X_train_f = pipeline.load_all_data(loaders)
    X_raw, Y_raw, X_raw_f = pipeline.load_all_data(loaders, kind='raw')
    X_test, Y_test, X_test_f = pipeline.load_all_data(loaders, kind='test')

    X_t, X_validate, y_t, y_validate = model_selection.train_test_split(X_train, Y_train, train_size=0.5)

    # plot train data with labels
    Plotter.plot_data(image_datasets, X_t, y_t, "Training Data with corresponding labels")
    # plot validate data with labels
    Plotter.plot_data(image_datasets, X_validate, y_validate, "Validation Data with corresponding labels")
    # plot test data with labels
    Plotter.plot_data(image_datasets, X_test, Y_test, "Test Data with corresponding labels")

    # Plot features extracted by SIFT
    pipeline.plot_features('data/train')

    # train model for Approach1
    model1 = pipeline.run_approach(1, X_train_f, X_train, X_raw_f, Y_train, image_datasets)
    # #get info for ROC
    # actuals, class_probabilities = pipeline.get_class_probabilities(model1, X_test_f, X_raw_f)
    # Plotter.plot_class_roc(actuals, class_probabilities)

    # # train model for Approach2
    model2 = pipeline.run_approach(2, X_train_f, X_train, X_test_f, Y_train, image_datasets)
    # # get info for ROC
    # actuals, class_probabilities = pipeline.get_image_probabilities(model2, X_test_f)
    # Plotter.plot_class_roc(actuals, class_probabilities)

    #models = pipeline.load_pretrained(".")
    # decide which model is better
    models = [model1, model2]
    # pipeline.save_models(".", models)
    #pipeline.render_test_data(models, X_test_f, X_raw_f)

    # corrupt all images with Gaussian noise
    sdevs = [0., 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    # calculate statistics
    for idx, model in enumerate(models):
        Fh, Ffa = pipeline.compute_statistics(model, X_test_f,  X_raw_f, idx+1)
        Plotter.plot_stats(Fh, Ffa, idx+1)

        stats, probs = pipeline.get_noise_stats(data_dir, model, sdevs, idx+1, X_raw_f)
        pd.DataFrame.from_dict(data=stats).to_csv(f'data-model{idx+1}.csv', header=False)

        Plotter.plot_noise_stats(stats, idx+1)
        Plotter.plot_noise_roc(probs, idx+1)




