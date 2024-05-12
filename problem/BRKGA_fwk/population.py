
import random

class Population:

    def __init__(self, config):
        self.config = config
        self.currentGeneration = [0] * config["numIndividuals"]
        for idx in range(config["numIndividuals"]):
            chromosome = [random.random() for _ in range(config["numGenes"])]
            self.currentGeneration[idx] = {'chr':chromosome, 'solution':{}, 'fitness':None}

    def createDeterministicIndividual(self):
        chromosome = [1] * self.config["numGenes"]
        return {'chr':chromosome, 'solution':None, 'fitness':None}

    def getGeneration(self):
        return self.currentGeneration

    def setGeneration(self, newGeneration):
        if newGeneration is None or len(newGeneration) != self.config["numIndividuals"]:
            raise Exception("ERROR: trying to set store a wrong generation vector")
        self.currentGeneration = newGeneration

    def setIndividual(self, individual, idx):
        if 0 > idx >= self.config["numIndividuals"]:
            raise Exception("ERROR: trying to set an individual in index: %s" % idx)
        self.currentGeneration[idx] = individual

    def classifyIndividuals(self):
        orderedGeneration = sorted(self.currentGeneration, key=lambda x: x['fitness'])
        elites=orderedGeneration[0:self.config["numElite"]]
        nonElites=orderedGeneration[self.config["numElite"]:len(self.currentGeneration)]
        return elites, nonElites

    def generateMutantIndividuals(self):
        mutants=[0] * self.config["numMutants"]
        for idx in range(self.config["numMutants"]):
            chromosome = [random.random() for _ in range(self.config["numGenes"])]
            mutants[idx] = {'chr':chromosome, 'solution':None, 'fitness':None}
        return mutants

    def doCrossover(self, elites, nonElites):
        crossover = [0] * self.config["numCrossover"]
        for idx in range(self.config["numCrossover"]):
            chrElite = random.choice(elites)['chr']
            chrNonElite = random.choice(nonElites)['chr']
            chrCross=[chrElite[gene] if random.random() <= self.config["inheritanceProb"] else chrNonElite[gene] for gene in range(self.config["numGenes"])]
            crossover[idx] = {'chr':chrCross, 'solution':None,'fitness':None}
        return crossover
