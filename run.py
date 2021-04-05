from PlantNN import *
from PlantAnalyzer import *
from DataLoader import * 


if __name__ == '__main__':
    analyzer = PlantAnalyzer(train_dir='plant-pathology-2021-fgvc8/train.csv')
    



    params = {'dim': 50,
          'batch_size': 32,
          'n_channels': 1,
          'shuffle': True,
          'n_classes':analyzer.n_classes,
          'train_dir':'plant-pathology-2021-fgvc8/train_images'}


    train_labels = dict(zip(analyzer.x_train, analyzer.y_train))
    val_labels = dict(zip(analyzer.x_test, analyzer.y_test))

    train_loader = DataLoader(analyzer.x_train,train_labels ,**params)
    val_loader = DataLoader(analyzer.x_test,val_labels ,**params)


    model_builder = PlantModel(dim=50, n_channels=1, optimizer='sgd', activation='relu', learning_rate_=0.001, n_classes=analyzer.n_classes)
    model = model_builder.build_model()
    model.fit_generator(generator=train_loader,
                    validation_data=val_loader,
                    epochs=30,
                   verbose=True)
