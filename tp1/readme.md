# TP1 PPAR - Erwan IQUEL & Mathieau LE CLEC'H

## Part 1: PRAM Emulation

1.A:  

```c
/*Au départ res = 0*/
for(i = 0; i < n; i++) {
  res += tab[i];
}
```

1.C:

```c
/* On somme les valeurs du tableau entre elles */
for(i = 0; i < (unsigned int) log2(n); i++) {
  for(j = 0; j < n; j++) {
    /* Si il y a un suivant alors on travail dessus */
    if(tab[j][1]) {
      tab[j][0] = tab[j][0] + tab[tab[j][1]][0];
      tab[j][1] = tab[tab[j][1]][1];
    }
  }
}
```

1.E:

```c
// finding the branch weights
/* On s'arrête une fois que tout les noeuds pointent sur le root ou qu'on a parcouru tout les noeuds */
for (i = 0; i < n && T[n-1] > 0; i++) {
  for(j = n - 1; j >= 0; j--) {
    /* On additionne nos valeurs afin d'avoir le poids total  */
    if(i == 0) {
      res[j] = val[j] + val[T[j]];
    } else {
      res[j] += val[T[j]];
    }

    if(T[j] != 0) {
      T[j] = T[T[j]];/* Déplacement */
    }
  }
}
```

## Part 2: Approaching the real parallelism: fork

We just call the fork function before the loop which calculate the branch weights


```c
/* Obligatoire pour utiliser la fonction fork */
#include <unistd.h>

...

// finding the branch weights
fork();
/* On s'arrête une fois que tout les noeuds pointent sur le root ou qu'on a parcouru tout les noeuds */
for (i = 0; i < n && T[n-1] > 0; i++) {
  for(j = n - 1; j >= 0; j--) {
    /* On additionne nos valeurs afin d'avoir le poids total  */
    if(i == 0) {
      res[j] = val[j] + val[T[j]];
    } else {
      res[j] += val[T[j]];
    }

    if(T[j] != 0) {
      T[j] = T[T[j]];/* Déplacement */
    }
  }
}
```

## Part 3: To sum up...

The complexity of differents algorithm

* 1.A:
  <table>
    <tr>
      <td>O(n)</<td>
    </tr>
  </table>
* 1.D:
  <table>
    <thead>
      <tr>
        <th>Sequential version</th>
        <th>Parallel version</th>
      </tr>
    </thead>
    
    <tbody>
      <tr>
        <td>n.log(n)</td>
        <td>log(n)</td>
      </tr>
    </tbody>
  </table>
* 1.E:
  <table>
    <thead>
      <tr>
        <th>Sequential version</th>
        <th>Parallel version</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>n.log(n)</td>
        <td>log(n)</td>
      </tr>
    </tbody>
  </table>
