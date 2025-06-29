with base as (
    select  order.id, order.name
    from orders order
    where order.id > 0
    union all
    select  order.id, order.name
    from orders as order
    inner join customers as customer
    on order.customer_id = customer.id
    where order.id > 5
),
base2 as (
    select  order.id, order.name
    from orders order inner join customers customer
    on order.customer_id = customer.id
    inner join products product
    on product.id = order.product_id
    where order.id > 0
)
select base.id, base.name
from base left join base2
on base.id = base2.id
left join product_ref
on product_ref.id = base.product_id
where base.id > 0
order by base.id, base.name
