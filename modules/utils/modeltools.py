from time import time

# Evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def compare_baseline_models(train, test, label, preprocessor, models):
    y_train = train[label]
    y_test = test[label]
    train_preprocessed = preprocessor.fit_transform(train)
    test_preprocessed = preprocessor.transform(test)

    for name, (model, params) in models.items():
        print("=============================")
        print(f"Evaluating [{name}] ...")
        print(f"\nParams: \n", model.get_params)

        start_time = time()
        model.fit(train_preprocessed, y_train)

        y_train_preds = model.predict(train_preprocessed)
        y_test_preds = model.predict(test_preprocessed)

        # metrics
        y_test_mse = mean_squared_error(y_test_preds, y_test)

        train_acc = model.score(train_preprocessed, y_train)
        train_acc = cross_val_score(model, train_preprocessed, y_train, cv=5, n_jobs=-1)
        test_acc = cross_val_score(model, test_preprocessed, y_test, cv=5, n_jobs=-1)

        print("\nMetrics:")
        print("test MSE = ", y_test_mse)
        print("train accuracy : ", train_acc)
        print("test accuracy : ", test_acc)
        print("time elapsed : ", time() - start_time)
        print("=============================")


def fine_tune_models(train, test, label, preprocessor, models):
    y_train = train[label]
    y_test = test[label]
    train_preprocessed = preprocessor.fit_transform(train)
    test_preprocessed = preprocessor.transform(test)

    for name, (model, params) in models.items():
        print("=============================")
        print(f"Fine-tunning [{name}] ...")
        start_time = time()

        if params != None:
            print(f"\nParams Grid:\n", params)

            gridSearch = GridSearchCV(
                model,
                params,
                n_jobs=-1,  # Use all cpus
                scoring="neg_mean_squared_error",
                return_train_score=True,
            )
            gridSearch.fit(train_preprocessed, y_train)

            model = gridSearch.best_estimator_
            models[name][0] = model

            print("\nGeneral Performance:")
            print("- mean_test_score:", gridSearch.cv_results_["mean_test_score"])
            print("\nPerformance of Best Model:")
            print("- Best score:", gridSearch.best_score_)
            print("- Best params:", gridSearch.best_params_)

            train_acc = cross_val_score(
                model, train_preprocessed, y_train, cv=5, n_jobs=-1
            )
            test_acc = cross_val_score(
                model, test_preprocessed, y_test, cv=5, n_jobs=-1
            )
            print("- Mean train accuracy:", train_acc.mean())
            print("- Mean Test accuracy:", test_acc.mean())

        print("\ntime elapsed : ", time() - start_time)
        print("=============================")
