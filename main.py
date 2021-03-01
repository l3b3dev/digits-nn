from Plotter import Plotter
from TrainingPipeline import TrainingPipeline

if __name__ == '__main__':
    data_dir = "data"

    pipeline = TrainingPipeline()
    image_datasets, loaders = pipeline.initialize_data(data_dir)
    X_train, Y_train, X_train_f = pipeline.load_all_data(loaders)
    X_raw, Y_raw, X_raw_f = pipeline.load_all_data(loaders, kind='raw')
    X_test, Y_test, X_test_f = pipeline.load_all_data(loaders, kind='test')

    # plot train data with labels
    Plotter.plot_data(image_datasets, X_train, Y_train)

    # train model for Approach1
    model1 = pipeline.run_approach(1, X_train_f, X_train, X_raw_f, Y_train, image_datasets)

    # # train model for Approach2
    model2 = pipeline.run_approach(2, X_train_f, X_train, X_test_f, Y_train, image_datasets)

    # decide which model is better
    models = [model1, model2]
    #pipeline.render_test_data(models, X_raw_f)
    # # #
    # # Model3 seems to be the best
    approach = 2
    model = models[approach - 1]

    # calculate statistics
    Fh, Ffa = pipeline.compute_statistics(model, X_raw_f, approach)

    Plotter.plot_stats(Fh, Ffa)

    # for_class = 5
    # actuals, class_probabilities = pipeline.get_class_probabilities(model, Y_test, X_test_f, for_class)
    # Plotter.plot_class_roc(actuals, class_probabilities, for_class)

    # for_class = 5
    actuals, class_probabilities = pipeline.get_image_probabilities(model, X_test_f, X_test_f)
    Plotter.plot_class_roc(actuals, class_probabilities, 1)

    # # corrupt all images with Gaussian noise
    # sdevs = [0., 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    # stats = pipeline.get_noise_stats(data_dir, model, sdevs, True)
    # #pd.DataFrame.from_dict(data=stats).to_csv('data.csv', header=False)
    #
    # Plotter.plot_noise_stats(stats)
