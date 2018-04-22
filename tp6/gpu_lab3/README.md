# Rendu TP2 GPGPU

Erwan IQUEL - Mathieu LE CLEC'H

---

## Game of Life

1. Version 1 de [life_kernel.cu](https://github.com/Olympus5/ppar/blob/c48ec65ba7ff4e067333ede57f034d651a8820c6/tp6/gpu_lab3/life_kernel.cu)

## Optimizing memory accesses

1. Il y a 9 lectures par case. Il n'y a pas de coalition:
    <table>
        <tr>
            <td>
                Voisin 1
            </td>
            <td>
                Voisin 2
            </td>
            <td>
                Voisin 3
            </td>
        </tr>
        <tr>
            <td>
                Voisin 4
            </td>
            <td>
                Case courante (Case problematique)
            </td>
            <td>
                Voisin 5
            </td>
        </tr>
        <tr>
            <td>
                Voisin 6
            </td>
            <td>
                Voisin 7
            </td>
            <td>
                Voisin 8
            </td>
        </tr>
    </table>

    Comme on peut voir ci-dessus, Nous n'avons pas *un seul bloc* contigu mais *deux blocs* contigus.
2. Si nous avons 1 seul bloc de 128 threads. Chaque threads devra lire au moins *128 * 8 (Ses cases voisines) + 128 (Sa case)* locations
3. Si nous avons des blocs à 2 dimensions (x >= 1, y >= 1 et z = 1), chaque threads devra lire 9 cases (lui puis ses voisins) (si on garde la même configuration pour les blocs qu'au départ). C'est donc la
première version qui diminue le nombre de relectures.
4. Version 2 de [life_kernel.cu]()

## Optimizing computations and datatypes

1. Le control flow (ou warp) divergence est ici dans le code kernel:
    ```c
    if(total < 2 ||total > 3 ||(myself == 0 && total != 3)) {
		myself = 0;
	} else {
		if(myself == 0) {
			myself = (red > blue) ? 1 : 2;
		}
	}
    ```
    Pour ce cas cela n'est pas possible de recoder afin de supprimer la divergence.
2. 29 registres sont utilisés.
3. 
