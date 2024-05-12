'''
AMMM Lab Heuristics
BRKGA solver
Copyright 2020 Luis Velasco.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Edited by Gianluca Graziadei 04/03/2024

'''

import time
from .population import Population


class Solver_BRKGA:

    def __init__(self, configuration,decoder):
        self.decoder = decoder
        self.population = Population(configuration)
        self.maxExecTime = configuration['maxExecTime']
        
    def stopCriteria(self):
        self.elapsedEvalTime = time.time() - self.startTime
        return self.elapsedEvalTime > self.maxExecTime

    def compare_fitness(self,fitness1,fitness2):
        if fitness1 is None:
            return True
        if fitness2 is None:
            return False
        
        j1,d1 = fitness1
        j2,d2 = fitness2
        return j1 > j2 or (j1 == j2 and d1 > d2)
    
    def solve(self, initial_solution=None, fitness=None):

        self.startTime = time.time()

        incumbent = self.population.createDeterministicIndividual()
        initialSolution = initial_solution

        if initialSolution is not None:
            incumbent['solution'] = initialSolution
            incumbent['fitness'] = fitness
            print("Initial solution with maximum jitter",fitness[0],"maximum delay",fitness[1])
            self.population.setIndividual(incumbent, 0)
        else:
            incumbent['solution'] = None
            incumbent['fitness'] = None

        generation = 0
        individualsDecoded = 0

        while True:
            generation += 1
            bestIndividual, numDecoded = self.decoder.decode(self.population.getGeneration())
            individualsDecoded += numDecoded
            
            if bestIndividual['fitness'] is None: break
            
            if self.compare_fitness(incumbent['fitness'],bestIndividual['fitness']):
                print("Generation",generation,"best fitness",bestIndividual['fitness'])
                incumbent = bestIndividual

            if self.stopCriteria(): break
            
            elites, nonElites = self.population.classifyIndividuals()
            mutants = self.population.generateMutantIndividuals()
            crossover = self.population.doCrossover(elites, nonElites)
            self.population.setGeneration(elites + crossover + mutants)

        self.numSolutionsConstructed = individualsDecoded
        return incumbent['solution']


