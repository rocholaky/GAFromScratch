import numpy as np
import random
#funciones que se utilizarán en los nodos intermedios
def node_sum(x, y):
    return x+y

def noide_multiply(x,y):
    return y*x

def node_divide(x, y):
    if y ==0:
        return 1
    else:
        return x/y

def node_subtract(x, y):
    return x - y




class Nodes:
    def __init__(self, component_element,level, var_representation, max_level =None):
        self.component = component_element
        self.var_representation = var_representation
        self.index = None #todo: check how to assign indexes
        self.level = level
        self.right = None
        self.left = None
        self.max_level = max_level

    def set_component(self, new_component):
        self.component = new_component

    def set_left(self, left_value):
        self.left = left_value

    def set_right(self, right_value):
        self.right = right_value

    def set_index(self, aIndex):
        self.index=aIndex

    def print_tree(self):
        if self.right==None:
            return self.component
        else:
            return '(' + self.left.print_tree() + self.component + self.right.print_tree() + ')'

    def print_order(self):
        if self.index == None:
            print('No order has yet been set')
        else:
            if self.right != None:
                return '{' + self.left.print_order() +'}'+ '['+str(self.index)+']' + '{'+ self.right.print_order()+'}'
            else:
                return '['+str(self.index)+']'

    def print_levels(self):
        if self.right == None:
            return '['+ str(self.level)+']'
        else:
            return '{'+ self.left.print_levels()+'}' + '['+str(self.level)+']' + '{'+self.right.print_levels()+'}'

    def evaluate(self, inputs, mode_chiffres= True):
        #assert len(inputs) == len(self.var_representation)
        if self.right == None:
            index = self.var_representation.index(self.component)
            return inputs[index]
        else:
            if self.component == '+':
                return self.left.evaluate(inputs, mode_chiffres) + self.right.evaluate(inputs, mode_chiffres)
            elif self.component == '*':
                return self.left.evaluate(inputs, mode_chiffres) * self.right.evaluate(inputs, mode_chiffres)
            elif self.component == '-':
                return self.left.evaluate(inputs, mode_chiffres) - self.right.evaluate(inputs, mode_chiffres)
            else:
                right_result = self.right.evaluate(inputs, mode_chiffres)
                left_result  = self.left.evaluate(inputs, mode_chiffres)

                if mode_chiffres:
                    if 0 == right_result:
                        return (right_result + 0.001)

                    else:
                        return left_result / right_result
                else:
                    if 0 in right_result:
                        return (right_result + 0.001)

                    else:
                        return left_result / right_result


    def copy_tree(self, with_index=True):
        if self.right == None:
            the_node = Nodes(self.component, self.level, self.var_representation, self.max_level)
            if with_index:
                the_node.set_index(self.index)
            return the_node
        else:
            the_node = Nodes(self.component, self.level, self.var_representation, self.max_level)
            if with_index:
                the_node.set_index(self.index)
            the_node.set_left(self.left.copy_tree())
            the_node.set_right(self.right.copy_tree())
            return the_node

    def get_tree_at_index(self, index):
        if self.index == index:
            return self.copy_tree()
        elif self.right == None:
            if self.index != index:
                print('index out of bounce')
                return -1
        else:
            if self.index < index:
                return self.right.get_tree_at_index(index)
            else:
                return self.left.get_tree_at_index(index)

    def change_tree_at_index(self, index, the_replacement):
        if self.index == index and self.level == 0:
            self.component = the_replacement.component
            self.set_left(the_replacement.left)
            self.set_right(the_replacement.right)
            return
        else:
            if self.right.index == index:
                self.set_right(the_replacement)
                return
            elif self.left.index == index:
                self.set_left(the_replacement)
                return
            else:
                if index<self.index:
                    self.left.change_tree_at_index(index, the_replacement)
                else:
                    self.right.change_tree_at_index(index, the_replacement)

    def replace_tree_at_index(self, index, replacement):
        new_tree = self.copy_tree()
        new_tree.change_tree_at_index(index, replacement)
        new_tree_no_index = new_tree.copy_tree(with_index=False)
        return new_tree_no_index

    def is_valid_index(self, level_for_insertion):
        if self.right == None:
            if self.distance_to_bottom()+ level_for_insertion<=self.max_level:
                return [self.index]
            else:
                return list()
        else:
            if self.distance_to_bottom() + level_for_insertion<=self.max_level:
                return self.left.is_valid_index(level_for_insertion) + [self.index] + self.right.is_valid_index(level_for_insertion)
            else:
                return self.left.is_valid_index(level_for_insertion) + self.right.is_valid_index(level_for_insertion)

    def possible_index(self, at_root = False):
        if self.level==1:
            if at_root:
                return self.left.possible_index() + [int(self.index)] + self.right.possible_index()
            else:
                return self.left.possible_index() + self.right.possible_index()

        elif self.right == None:
            return [int(self.index)]
        else:
            return self.left.possible_index() + [int(self.index)] + self.right.possible_index()



    def get_level(self, index):
        if self.index == index:
            return self.level
        else:
            if self.index > index:
                return self.left.get_level(index)
            else:
                return self.right.get_level(index)



    def max_index(self):
        if self.right == None:
            return self.index
        else:
            return self.right.max_index

    def distance_to_bottom(self):
        if self.right == None:
            return 0
        else:
            return 1+ max(self.right.distance_to_bottom(), self.left.distance_to_bottom())

    def distance_to_bottom_from_index(self, index):
        if self.index == index:
            return self.distance_to_bottom()
        else:
            if self.right==None:
                return
            elif self.index>index:
                return self.left.distance_to_bottom_from_index(index)
            else:
                return self.right.distance_to_bottom_from_index(index)

    def set_level(self, alevel):
        self.level = alevel








 ########################### librería tree ##############################3

class tree_factory:
    def __init__(self, inputs, depth):
        # define the depth, the function possibilities, the inputs and the maximum level that a tree can have:
        self.depth = int(depth)
        self.internal_function_posibility = ['+', '-', '/', '*']
        self.inputs = inputs
        self.max_level = self.depth


    def generate_tree(self, level= 1):
        if level <= self.depth:
            tree = Nodes(random.choice(self.internal_function_posibility), level=level, var_representation = self.inputs,\
                         max_level=self.depth+1)
            tree.set_left(self.generate_tree(level+1))
            tree.set_right(self.generate_tree(level + 1))
            return tree
        else:
            tree = Nodes(random.choice(self.inputs), level=level, var_representation = self.inputs,\
                         max_level=self.depth+1)
            tree.set_left(None)
            tree.set_right(None)
            return tree

    def set_tree_index(self, atree, prev_index=None):
        if prev_index == None:
            if atree.right != None:
                left_index = self.set_tree_index(atree.left, prev_index=None)
                atree.set_index(left_index + 1)
                right_index = self.set_tree_index(atree.right, atree.index)
                return right_index
            else:
                atree.set_index(1)
                return 1

        else:
            if atree.right == None:
                atree.set_index(prev_index+1)
                return prev_index+1
            else:
                left_index = self.set_tree_index(atree.left, prev_index=prev_index)
                atree.set_index(left_index+1)
                right_index = self.set_tree_index(atree.right, atree.index)
                return right_index

    def set_tree_levels(self, atree, prev_level=0):
        if atree.right == None:
            atree.set_level(prev_level+1)
        else:
            atree.set_level(prev_level+1)
            self.set_tree_levels(atree.left, prev_level=prev_level+1)
            self.set_tree_levels(atree.right, prev_level=prev_level + 1)



