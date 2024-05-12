'''
AMMM Lab Heuristics
Abstract Decoder class
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

class _Decoder:

    def decode(self, generation):
        numDecoded = 0
        bestInGeneration = {'chr':None, 'solution':None, 'fitness':None}
        for individual in generation:
            numDecoded += 1
            if individual['fitness'] is None:
                solution,j,d,d_fi,j_f = self.decodeIndividual(individual['chr'])
                if solution is None: continue
                individual['solution'] = solution,d_fi,j_f
                individual['fitness'] = j,d

                if not bestInGeneration['fitness']:
                    bestInGeneration = individual
                else:
                    j,d = individual['fitness']
                    _j,_d = bestInGeneration['fitness']

                    if j<_j or (j==_j and d<_d):
                        bestInGeneration = individual   

        return bestInGeneration, numDecoded

    def decodeIndividual(self, chromosome):
        raise NotImplementedError