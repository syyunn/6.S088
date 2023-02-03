  # load data
  import numpy as np
  import torch
  import torchvision
  import torchvision.transforms as transforms

  # transform = transforms.Compose(
  #     [transforms.ToTensor(),
  #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True)
  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True)

  label_map = {"dog": 5, "cat": 3}

  dog_cat_train = [trainset[i][1] == 5 or trainset[i][1] == 3 for i in range(len(trainset))]
  dog_cat_test = [testset[i][1] == 5 or testset[i][1] == 3 for i in range(len(testset))]


  train = trainset.data[dog_cat_train]
  train_label = np.array(trainset.targets)[dog_cat_train]
  test = testset.data[dog_cat_test]
  test_label = np.array(testset.targets)[dog_cat_test]

  import matplotlib.pyplot as plt
  import numpy as np

  # functions to show an image

  def imshow(img):
      # img = img / 2 + 0.5     # unnormalize
      # plt.imshow(np.transpose(img, (1, 2, 0)))
      plt.imshow(img)
      plt.show()

  # images, labels = train[0], train_label[0]
  # imshow(images)

  # check sanity
  print(train.shape)
  print(train_label.shape)
  print(test.shape)
  print(test_label.shape)

  # flatten 
  X_train = train.reshape(train.shape[0], -1)
  X_test = test.reshape(test.shape[0], -1)
  print(X_train.shape)
  print(X_test.shape)

  # flatten labels
  Y_train = train_label.reshape(train_label.shape[0], -1)
  Y_test = test_label.reshape(test_label.shape[0])

  import neural_tangents as nt
  from neural_tangents import stax

  K = [1, 3, 5, 7]
  k = 1 # number of hidden layers
  layers = []
  for _ in range(k):
    layers += [stax.Dense(3072), stax.Relu()]
  layers += [stax.Dense(1)]

  _, _, kernel_fn = stax.serial(*layers)

  predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, 
                                                        X_train,
                                                        Y_train)

  Y_test_pred = predict_fn(x_test=X_test, get='ntk')

  Y_test_pred = [5 if i > 4 else 3 for i in Y_test_pred]

  accuracy = np.sum(Y_test_pred == Y_test) / len(Y_test)

  if __name__ == '__main__':
      pass
