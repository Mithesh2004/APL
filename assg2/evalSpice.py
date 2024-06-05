import numpy as np
import io

def addVsource(vElement, vlst, nodes):
    dic = {}  # here i am storing each voltage source in a dictionary
    dic["name"] = vElement[0]

    # dic[n1] will be equal to 0 if vElement[1] == "GND" else it will the integer present in vElement[1](which is node 1)
    dic["n1"] = (
        0
        if vElement[1] == "GND"
        else int(
            "".join(x for x in vElement[1] if x.isdigit())
        )  # taking only the integers present in vElement[1]
    )
    # dic[n2] will be equal to 0 if vElement[2] == "GND" else it will the integer present in vElement[2](which is node 2)
    dic["n2"] = (
        0
        if vElement[2] == "GND"
        else int("".join(x for x in vElement[2] if x.isdigit()))
    )
    dic["type"] = vElement[3]
    dic["val"] = float(vElement[4])
    
    #appending the nodes in a list
    if vElement[1] not in nodes:
        nodes.append(vElement[1])
    if vElement[2] not in nodes:
        nodes.append(vElement[2])
    vlst.append(dic)
    return vlst, nodes


def addIsource(iElement, ilst, nodes):
    dic = {}
    dic["n1"] = (
        0
        if iElement[1] == "GND"
        else int("".join(x for x in iElement[1] if x.isdigit()))
    )
    dic["n2"] = (
        0
        if iElement[2] == "GND"
        else int("".join(x for x in iElement[2] if x.isdigit()))
    )
    dic["type"] = iElement[3]
    dic["val"] = float(iElement[4])
    if iElement[1] not in nodes:
        nodes.append(iElement[1])
    if iElement[2] not in nodes:
        nodes.append(iElement[2])
    ilst.append(dic)
    return ilst, nodes


def addResistance(rElement, rlst, nodes):
    dic = {}
    dic["n1"] = (
        0
        if rElement[1] == "GND"
        else int("".join(x for x in rElement[1] if x.isdigit()))
    )
    dic["n2"] = (
        0
        if rElement[2] == "GND"
        else int("".join(x for x in rElement[2] if x.isdigit()))
    )
    if float(rElement[3]) == 0.0: 
        dic["val"] = 10 ** (-10)    # if the resistance of the wire is 0, i am alloting a low value of 10^(-10) to the resistor
    elif float(rElement[3]) < 0:
        raise ValueError("Resistace cannot be negative") 
    else:
        dic["val"] = float(rElement[3])
        
    if rElement[1] not in nodes:
        nodes.append(rElement[1])
    if rElement[2] not in nodes:
        nodes.append(rElement[2])
        
    rlst.append(dic)
    return rlst, nodes


def extractFile(filename):
    with open(filename) as f:
        data = f.read()
    sfp = io.StringIO(data)
    lst = sfp.readlines()
    newlst = [x[: x.index("#")] if "#" in x else x for x in lst]  # considering only the string before '#' if it is present(removing the comments)
    extractedlst = [x.split() for x in newlst]
    while [] in extractedlst:
        extractedlst.remove([]) # if a comment is present in a newline it will give an empty list in extractedlst 
    return extractedlst 

def processFile(extractedlst):
    vlst = []
    rlst = []
    ilst = []
    nodes = []
    if [".circuit"] and [".end"] in extractedlst:
        startIndex = extractedlst.index([".circuit"]) + 1
        endIndex = extractedlst.index([".end"])
    else:
        raise ValueError("Malformed circuit file")
    elements = extractedlst[startIndex:endIndex] #removing [".circuit"] and [".end"] from the extracted list
    for element in elements:
        elementType = element[0]
        if elementType[0] == "V":
            if len(element) == 5:  # as per the input file the no. of elements for a vsource should be 5 which is [v,n1,n2,dc,val]
                vlst, nodes = addVsource(element, vlst, nodes)
            else:
                raise ValueError("Malformed circuit file")
        elif elementType[0] == "I":
            if len(element) == 5:
                ilst, nodes = addIsource(element, ilst, nodes)
            else:
                raise ValueError("Malformed circuit file")

        elif elementType[0] == "R":
            if len(element) == 4:
                rlst, nodes = addResistance(element, rlst, nodes)
            else:
                raise ValueError("Malformed circuit file")

        else:
            raise ValueError("Only V, I, R elements are permitted")
    nodes.sort()
    return vlst, ilst, rlst, nodes


def createMatrix(vlst, ilst, rlst, nodes):
    num_nodes = len(nodes) - 1
    num_vsources = len(vlst)
    A = np.zeros((num_nodes + num_vsources, num_nodes + num_vsources)) 
    B = np.zeros((num_nodes + num_vsources, 1))
    arrsize = num_nodes + num_vsources - 1
    
    #nodal analysis
    for element in rlst:
        n1 = element["n1"]
        n2 = element["n2"]
        val = element["val"]

        if n1 != 0:
            A[n1 - 1, n1 - 1] += 1 / val
            if n2 != 0:
                A[n1 - 1, n2 - 1] -= 1 / val
                A[n2 - 1, n1 - 1] -= 1 / val
                A[n2 - 1, n2 - 1] += 1 / val
            B[n1 - 1] = 0

        elif n2 != 0:
            A[n2 - 1, n2 - 1] += 1 / val
            B[n2 - 1] = 0

    for element in vlst:
        n1 = element["n1"]
        n2 = element["n2"]
        val = element["val"]

        if n1 != 0:
            A[n1 - 1, arrsize] = 1
            A[arrsize, n1 - 1] = 1
            B[arrsize] = val

        elif n2 != 0:
            A[n2 - 1, arrsize] = -1
            A[arrsize, n2 - 1] = -1
            B[arrsize] = val

    for element in ilst:
        n1 = element["n1"]
        n2 = element["n2"]
        val = element["val"]

        if n1 != 0:
            B[n1 - 1] = -val

        elif n2 != 0:
            B[n2 - 1] = val

    return A, B


def evalSpice(filename):
    try:
        extractedlst = extractFile(filename)
        (vlst, ilst, rlst, nodes) = processFile(extractedlst)
        A, B = createMatrix(vlst, ilst, rlst, nodes)
        x = np.linalg.solve(A, B)
        vdict = {"GND": 0.0}
        if "GND" in nodes:
            nodes.remove("GND")
        elif "0" in nodes:
            nodes.remove(0)
        for velement in vlst:
            idict = {velement["name"]: x[-1][0]} #taking the value of last element in x (which is the current through vsource)
        for i in range(len(nodes)):
            vdict[nodes[i]] = x[i][0] # mapping respective node voltages with its value
        return (vdict, idict)

    except np.linalg.LinAlgError:   #exception to handle if the A matrix is non-invertible
        raise ValueError("Circuit error: no solution")

    except FileNotFoundError:
        raise FileNotFoundError("Please give the name of a valid SPICE file as input")