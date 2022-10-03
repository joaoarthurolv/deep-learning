import numpy as np
# transfer function

# problemas linearmente separáveis
def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

# problemas de classificação binária
def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))

# classificação - -1 e 1
def tanFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

# redes convolucionais e redes com muitas camadas
def reluFunction(soma):
    if(soma >= 0):
        return soma
    return 0

# regressão
def linearFunction(soma):
    return soma

# classificação com mais de 2 classes
def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()


teste = stepFunction(-1)
teste = sigmoidFunction(0.358)
teste = tanFunction(-0.358)
teste = reluFunction(0.358)
teste = linearFunction(-0.358)

valores = [7.0, 2.0, 1.3]

print(softmaxFunction(valores))