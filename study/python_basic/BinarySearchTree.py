"""
Reference. https://gist.github.com/jakemmarsh/8273963
"""

outputdebug = False


def debug(msg):
    if outputdebug:
        print(msg)


class Node(object):
    def __init__(self, val):
        self.val = val
        self.leftChild = None
        self.rightChild = None

    def get(self):
        return self.val

    def set(self, val):
        self.val = val

    def getChildren(self):
        children = []
        if self.leftChild is not None:
            children.append(self.leftChild)
        if self.rightChild is not None:
            children.append(self.rightChild)
        return children


class BST(object):
    def __init__(self):
        self.root = None

    def setRoot(self, val):
        self.root = Node(val)

    def insert(self, val):
        if self.root is None:
            self.setRoot(val)
        else:
            self.insertNode(self.root, val)

    def insertNode(self, currentNode, val):
        if val <= currentNode.val:
            if currentNode.leftChild:
                self.insertNode(currentNode.leftChild, val)
            else:
                currentNode.leftChild = Node(val)
        elif val > currentNode.val:
            if currentNode.rightChild:
                self.insertNode(currentNode.rightChild, val)
            else:
                currentNode.rightChild = Node(val)

    def find(self, val):
        return self.findNode(self.root, val)

    def findNode(self, currentNode, val):
        if currentNode is None:
            return False
        elif val == currentNode.val:
            return True
        elif val < currentNode.val:
            return self.findNode(currentNode.leftChild, val)
        else:
            return self.findNode(currentNode.rightChild, val)

    def delete(self, val):
        if self.root is None:
            return None
        else:
            self.deleteNode(self.root, val)

    def findPrecessorNode(self, currentNode):
        """
        Find the bigger valued node in RIGHT child
        """
        node = currentNode.leftChild
        # just a sanity check
        if node is not None:

            while node.rightChild is not None:
                debug("RS: traversing: " + str(node.val))
                if node.rightChild is None:
                    return node
                else:
                    node = node.rightChild
        return node

    def findSuccessorNode(self, currentNode):
        """
        Find the smaller valued node in RIGHT child
        """
        node = currentNode.rightChild
        # just a sanity check
        if node is not None:
            while node.leftChild is not None:
                debug("LS: traversing: " + str(node.val))
                if node.leftChild is None:
                    return node
                else:
                    node = node.leftChild
        return node

    def deleteNode(self, currentNode, val):
        LeftChild = currentNode.leftChild
        RightChild = currentNode.rightChild
        debug(f"current value: {currentNode.val},target: {val}")

        # Value Check
        if currentNode is None:
            return False

        if currentNode.val == val:
            currentChild = currentNode.getChildren()
            if len(currentChild) == 0:
                debug("Root, Non-Child case")
                currentNode = None
                return True
            elif len(currentChild) == 1:
                debug("Root, a Child case")
                currentNode = currentChild[0]
                return True
            else:
                debug("Root, two children case")
                successorNode = self.findSuccessorNode(currentNode)
                successorValue = successorNode.val
                self.delete(successorValue)
                currentNode.val = successorValue
                return True

        if LeftChild is not None:
            if val == LeftChild.val:
                LeftChildChild = LeftChild.getChildren()
                if len(LeftChildChild) == 0:
                    debug("Left, Non-Child case")
                    currentNode.leftChild = None
                    return True
                elif len(LeftChildChild) == 1:
                    debug("Left, a Child case")
                    currentNode.leftChild = LeftChildChild[0]
                    return True
                else:
                    debug("Left, two children case")
                    successorNode = self.findSuccessorNode(LeftChild)
                    successorValue = successorNode.val
                    self.delete(successorValue)
                    currentNode.leftChild.val = successorValue
                    return True
            else:
                pass

        if RightChild is not None:
            if val == RightChild.val:
                RightChildChild = RightChild.getChildren()
                if len(RightChildChild) == 0:
                    debug("Right, Non-Child case")
                    currentNode.rightChild = None
                    return True
                elif len(RightChildChild) == 1:
                    debug("Right, a Child case")
                    currentNode.rightChild = RightChildChild[0]
                    return True
                else:
                    debug("Right, two children case")
                    successorNode = self.findSuccessorNode(RightChild)
                    successorValue = successorNode.val
                    self.delete(successorValue)
                    currentNode.rightChild.val = successorValue
                    return True
            else:
                pass

        # Move Child Node
        if val < currentNode.val:
            debug("Go to Left")
            return self.deleteNode(currentNode.leftChild, val)
        else:
            debug("Go to Right")
            return self.deleteNode(currentNode.rightChild, val)

    def traverse(self):
        return self.traverseNode(self.root)

    def traverseNode(self, currentNode):
        result = []
        if currentNode.leftChild is not None:
            result.extend(self.traverseNode(currentNode.leftChild))
        if currentNode is not None:
            result.extend([currentNode.val])
        if currentNode.rightChild is not None:
            result.extend(self.traverseNode(currentNode.rightChild))
        return result


# Usage example
if __name__ == "__main__":
    a = BST()
    print("----- Inserting -------")
    # inlist = [5, 2, 12, -4, 3, 21, 19, 25]
    inlist = [7, 5, 2, 6, 3, 4, 1, 8, 9, 0]
    for i in inlist:
        a.insert(i)

    print(a.traverse())
    import copy

    print("----- Deleting -------")
    test = copy.deepcopy(a)
    del_list = test.traverse()
    for value in del_list:
        print(f"delete {value}")
        test.delete(value)
        print(test.traverse())
        test = copy.deepcopy(a)
