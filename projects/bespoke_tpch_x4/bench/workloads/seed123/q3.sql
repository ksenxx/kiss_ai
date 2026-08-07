select l_orderkey,  
    sum(l_extendedprice*(1-l_discount)) as revenue, 
    o_orderdate,  
    o_shippriority 
FROM
    customer,  
    orders,  
    lineitem 
WHERE
    c_mktsegment = 'FURNITURE' 
    and c_custkey = o_custkey 
    and l_orderkey = o_orderkey 
    and o_orderdate < date '1995-03-04' 
    and l_shipdate > date '1995-03-04' 
GROUP BY
    l_orderkey,  
    o_orderdate,  
    o_shippriority 
ORDER BY
    revenue desc,  
    o_orderdate;